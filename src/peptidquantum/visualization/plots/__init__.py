from .contact_map import ContactMapPlotter

__all__ = [
    "ContactMapPlotter",
    "generate_site_extra_assets",
    "html_extra_viz_section",
]


def __getattr__(name: str):
    if name in ("generate_site_extra_assets", "html_extra_viz_section"):
        from . import site_extras

        return getattr(site_extras, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
