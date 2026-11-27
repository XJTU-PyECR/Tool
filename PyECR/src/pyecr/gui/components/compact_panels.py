"""Responsive presentation helpers; never read or commit scientific inputs."""


def content_width(dpg, item):
    parent = dpg.get_item_parent(item)
    while parent:
        if dpg.get_item_info(parent)['type'].endswith(('mvChildWindow', 'mvWindowAppItem')):
            size = dpg.get_item_state(parent).get('rect_size', (0, 0))[0]
            if size > 0:
                return max(240, int(size) - 40)
        parent = dpg.get_item_parent(parent)
    return max(240, dpg.get_viewport_client_width() - 80)


def resize_pair(dpg, row, panels, *, minimum=340):
    if not dpg.does_item_exist(row):
        return
    width = content_width(dpg, row)
    horizontal = width >= 2 * minimum + 12
    column = (width - 12) // 2 if horizontal else width
    dpg.configure_item(row, horizontal=horizontal)
    for panel in panels:
        dpg.configure_item(panel, width=column)
        queue = list(dpg.get_item_children(panel, 1) or [])
        while queue:
            item = queue.pop()
            kind = dpg.get_item_info(item)['type'].split('::')[-1]
            if kind == 'mvText':
                dpg.configure_item(item, wrap=max(100, column - 20))
            if kind not in ('mvPlot', 'mvTable'):
                queue.extend(dpg.get_item_children(item, 1) or [])


def field(dpg, title, create, **kwargs):
    """Table cell with a full caption and a fill-width value."""
    with dpg.group():
        dpg.add_text(title)
        return create(width=-1, **kwargs)
