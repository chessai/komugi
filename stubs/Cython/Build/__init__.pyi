from typing import Any

def cythonize(
    module_list: Any,
    exclude: Any = ...,
    nthreads: int = ...,
    aliases: Any = ...,
    quiet: bool = ...,
    force: bool = ...,
    language: str | None = ...,
    exclude_failures: bool = ...,
    show_all_warnings: bool = ...,
    compiler_directives: dict[str, Any] | None = ...,
    **options: Any,
) -> list[Any]: ...
