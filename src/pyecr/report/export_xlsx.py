"""Portable optional Python XLSX backend; scientific machine tables stay exact."""
from math import isfinite
from math import ceil
import json

EXCEL_MIN = 2.2251e-308
EXCEL_MAX = 9.99999999999999e307


def write_xlsx(tables, destination, options, overview, check=lambda: None):
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError("Excel export requires the existing pyecr[xlsx] optional dependency") from exc
    workbook = Workbook(write_only=True)
    workbook.iso_dates = True
    issues, sheets = [], []
    columns_by_sheet, next_row_by_sheet = {}, {}
    body = Font(name="Arial", size=10)
    heading = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="244E70")

    def new_sheet(name, columns, units):
        sheet = workbook.create_sheet(name)
        columns_by_sheet[name], next_row_by_sheet[name] = columns, 2
        sheet.freeze_panes = "A2"
        sheet.sheet_view.showGridLines = False
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.paperSize = "9"  # OOXML A4; streaming worksheet has no constants.
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        header = []
        for i, col in enumerate(columns, 1):
            label = f"{col} [{units[col]}]" if col in units else col
            cell = WriteOnlyCell(sheet, label)
            cell.font, cell.fill = heading, fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            header.append(cell)
            width = 28 if col not in {"reason", "fixed_source", "ci_conditions", "path", "value_json"} else 55
            sheet.column_dimensions[get_column_letter(i)].width = width
        sheet.row_dimensions[1].height = 36
        sheet.append(header)
        return sheet

    def cells(sheet, values, kinds, table, row):
        result = []
        for index, (value, kind) in enumerate(zip(values, kinds, strict=True), 1):
            cell = WriteOnlyCell(sheet)
            cell.font = body
            cell.alignment = Alignment(horizontal="right" if kind.startswith(("float", "integer")) else "left", vertical="center", wrap_text=kind.startswith(("text","json")))
            if value is None:
                cell.value = None
            elif kind.startswith("float"):
                numeric = float(value)
                if not isfinite(numeric) or (numeric != 0 and abs(numeric) < EXCEL_MIN) or abs(numeric) > EXCEL_MAX:
                    cell.value, cell.data_type = "n.a.", "s"
                    issues.append((table, row, index, "unavailable_in_excel_numeric_range", f"tables/{table}.csv"))
                else:
                    cell.value = float(format(numeric, ".15g"))
                    cell.number_format = "0.00000000000000E+00"
            elif kind.startswith("integer"):
                cell.value = int(value)
                cell.number_format = "0"
            elif kind.startswith("boolean"):
                cell.value = bool(value)
            else:
                text = str(value)
                invalid_xml = any(ord(c) < 32 and c not in "\t\n\r" for c in text)
                if invalid_xml or len(text) > 200:
                    cell.value = f"See tables/{table}.csv, data row {row}, column {index}"
                    issues.append((table, row, index, "unsupported_excel_control_character" if invalid_xml else "long_value_in_scientific_file", f"tables/{table}.csv"))
                else:
                    # Assign explicit string type after value to defeat '=' auto-formulas.
                    cell.value = text
                cell.data_type = "s"
                cell.number_format = "@"
            result.append(cell)
        lines = max((sum(max(1,ceil(len(line)/(55 if index in {"reason","fixed_source","ci_conditions","path","value_json"} else 28))) for line in str(c.value).splitlines())
                     for index,c in zip(columns_by_sheet[sheet.title],result) if c.data_type=="s" and c.value is not None),default=1)
        if lines>1:
            sheet.row_dimensions[next_row_by_sheet[sheet.title]].height=min(409,lines*14+4)
        next_row_by_sheet[sheet.title] += 1
        return result

    try:
        check()
        summary = new_sheet("Overview", ("item", "value"), {})
        for i, (key, value) in enumerate(overview.items(), 1):
            summary.append(cells(summary, (key, str(value)), ("text", "text"), "Overview", i))
        summary.auto_filter.ref = f"A1:B{len(overview)+1}"
        for table in tables:
            check()
            part, offset, written = 1, 0, 0
            sheet = new_sheet(table.name, table.columns, table.units)
            for row in table.rows():
                if written and written % options.max_data_rows_per_sheet == 0:
                    sheet.auto_filter.ref = f"A1:{get_column_letter(len(table.columns))}{written-offset+1}"
                    sheets.append({"table": table.name, "sheet": sheet.title, "start_index": offset, "rows": written-offset})
                    offset, part = written, part+1
                    sheet = new_sheet(f"{table.name}_{part:03d}", table.columns, table.units)
                if written % 1024 == 0:
                    check()
                sheet.append(cells(sheet, row, table.types, table.name, written+1))
                written += 1
            if written != table.row_count:
                raise ValueError(f"XLSX {table.name} row count mismatch")
            sheet.auto_filter.ref = f"A1:{get_column_letter(len(table.columns))}{written-offset+1}"
            sheets.append({"table": table.name, "sheet": sheet.title, "start_index": offset, "rows": written-offset})
        dictionary = new_sheet("DataDictionary", ("table", "field", "type", "unit", "source"), {})
        for table in tables:
            for col, kind in zip(table.columns, table.types, strict=True):
                dictionary.append(cells(dictionary, (table.name,col,kind,table.units.get(col,""),f"tables/{table.name}.csv"), ("text",)*5, "DataDictionary", 1))
        statuses = new_sheet("WorkbookStatus", ("table", "data_row", "column", "status", "scientific_source"), {})
        for row in issues:
            statuses.append(cells(statuses, row, ("text","integer","integer","text","text"), "WorkbookStatus", 1))
        check()
        workbook.save(destination)
        check()
        return {"sheets": sheets, "precision_digits": 15, "numeric_cells_are_approximate": True,
                "scientific_authority": "analysis.json and 17g CSV", "issues": [list(i) for i in issues]}
    except BaseException:
        # openpyxl streaming sheets own temporary XML files. Close/clean these
        # as well as the exporter-owned directory on cooperative abort.
        for sheet in workbook.worksheets:
            writer = getattr(sheet, "_writer", None)
            if writer is not None:
                try:
                    if not sheet.closed:
                        sheet.close()
                    writer.cleanup()
                except (OSError, ValueError):
                    pass
        raise
    finally:
        workbook.close()


def verify_xlsx_tables(tables, destination, info, check=lambda: None):
    """Read actual XLSX cells against captured tables using existing P71 rules."""
    from math import isclose
    from itertools import islice
    from openpyxl import load_workbook
    wb = load_workbook(destination, read_only=True, data_only=False)
    try:
        table_map = {t.name:t for t in tables}
        covered = {t.name:0 for t in tables};seen_tables=set();seen_sheets=set()
        from pyecr.report.export_xlsx import EXCEL_MIN, EXCEL_MAX
        for part in info["sheets"]:
            check()
            if (part['table'] not in table_map or part['sheet'] in seen_sheets
                    or type(part['start_index']) is not int or type(part['rows']) is not int
                    or part['rows']<0 or part['start_index']!=covered[part['table']]):
                raise ValueError('XLSX table partition identity/coverage mismatch')
            seen_tables.add(part['table']);seen_sheets.add(part['sheet'])
            sheet = wb[part["sheet"]]
            table = table_map[part["table"]]
            cells = sheet.iter_rows()
            header = [c.value for c in next(cells)]
            expected_header = [f"{c} [{table.units[c]}]" if c in table.units else c for c in table.columns]
            if header != expected_header:
                raise ValueError("XLSX field/unit header mismatch")
            count = 0
            expected_rows = islice(table.rows(), part["start_index"], part["start_index"]+part["rows"])
            for native, expected in zip(cells, expected_rows, strict=True):
                if count % 1024 == 0:
                    check()
                for i,(cell, original, kind) in enumerate(zip(native, expected, table.types, strict=True),1):
                    if cell.data_type == "f":
                        raise ValueError("XLSX contains unexpected formula")
                    actual = cell.value
                    if original is None:
                        valid = actual is None
                    elif kind.startswith("float"):
                        n = float(original)
                        if not isfinite(n) or (n!=0 and abs(n)<EXCEL_MIN) or abs(n)>EXCEL_MAX:
                            valid = actual == "n.a."
                        else:
                            valid = isinstance(actual,(int,float)) and not isinstance(actual,bool) and isclose(float(actual),float(format(n,".15g")),rel_tol=5e-15,abs_tol=0.)
                    elif kind.startswith(("integer","boolean")):
                        valid = actual == original
                    else:
                        text = str(original)
                        invalid = any(ord(c)<32 and c not in "\t\n\r" for c in text)
                        expected_text = f"See tables/{table.name}.csv, data row {part['start_index']+count+1}, column {i}" if invalid or len(text)>200 else text
                        valid = actual == expected_text or (expected_text == "" and actual is None)
                    if not valid:
                        raise ValueError(f"XLSX precision/type/value mismatch: {table.name} row {count} field {table.columns[i-1]}")
                count += 1
            if count != part["rows"]:
                raise ValueError("XLSX real-file row count mismatch")
            covered[part['table']]+=count
        if seen_tables!=set(table_map) or any(covered[k]!=v.row_count for k,v in table_map.items()):
            raise ValueError('XLSX incomplete scientific table coverage')
    finally:
        wb.close()
