"""Griffe extension that turns the Sphinx roles of the docstrings into MkDocs cross-references.

``:func:`name``` and ``:class:`~package.Name``` become ``[`name`][path]`` and ``[`Name`][path]``.
Targets are looked up from the documented object outwards, as in Sphinx. A function re-exported
under the name of its module (``opticalflow_torch.farneback``) resolves to the function. Other
targets are left to the Sphinx inventories configured in ``mkdocs.yml``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(func|class|meth|attr|data|obj|exc|mod):`(~?)([\w.]+)`")

# Names that the inventories list under another path.
_EXTERNAL = {"torch.nn.Parameter": "torch.nn.parameter.Parameter"}


def _objects(obj: griffe.Object) -> Iterator[griffe.Object]:
    yield obj
    for member in obj.members.values():
        if not member.is_alias:
            yield from _objects(member)


def _lookup(root: griffe.Module, path: str, role: str) -> str | None:
    """Canonical path of ``path``, or the target path of an alias to another package."""
    try:
        found = root[path.split(".", 1)[1]] if "." in path else root
        if found.is_alias:
            if not found.target_path.startswith(f"{root.name}."):
                return found.target_path
            found = found.final_target
        if found.is_module and role != "mod":
            found = found.members[found.name]  # the object re-exported over its module
            if found.is_alias:
                found = found.final_target
    except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return None
    return found.path


def _resolve(target: str, obj: griffe.Object, role: str) -> str:
    root = obj.package
    if target.split(".")[0] == root.name:
        scopes = [target]
    else:
        scopes, scope = [], obj
        while scope is not None:
            scopes.append(f"{scope.path}.{target}")
            scope = scope.parent
    for path in scopes:
        found = _lookup(root, path, role)
        if found is not None:
            return found
    return _EXTERNAL.get(target, target)


class SphinxRoles(griffe.Extension):
    """Rewrite Sphinx cross-reference roles once the package is loaded."""

    def on_package(self, *, pkg: griffe.Module, **kwargs: Any) -> None:
        for obj in _objects(pkg):
            if obj.docstring is None:
                continue

            def link(match: re.Match[str], obj: griffe.Object = obj) -> str:
                role, short, target = match.groups()
                text = target.rsplit(".", 1)[-1] if short else target
                return f"[`{text}`][{_resolve(target, obj, role)}]"

            obj.docstring.value = _ROLE.sub(link, obj.docstring.value)
