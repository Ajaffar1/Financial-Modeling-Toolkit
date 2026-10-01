def circular_references(dependencies):
    """Return cycle paths in an explicitly supplied dependency graph."""
    visited, active, path, cycles = set(), set(), [], []
    def visit(node):
        if node in active:
            cycles.append(tuple(path[path.index(node):] + [node]))
            return
        if node in visited:
            return
        active.add(node)
        path.append(node)
        for child in dependencies.get(node, ()):
            visit(child)
        path.pop()
        active.remove(node)
        visited.add(node)
    for node in dependencies:
        visit(node)
    return cycles
