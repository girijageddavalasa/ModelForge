"""Model plugin registry."""

from __future__ import annotations

from app.ml.base_plugin import ModelPlugin


class PluginRegistry:
    """Register and resolve model plugins by stable name."""

    def __init__(self) -> None:
        self._plugins: dict[str, ModelPlugin] = {}

    def register(self, plugin: ModelPlugin) -> None:
        """Register a plugin and reject duplicate names."""
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name!r} is already registered.")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> ModelPlugin:
        """Return a plugin or raise a clear lookup error."""
        try:
            return self._plugins[name]
        except KeyError as error:
            raise LookupError(f"Unknown model plugin: {name}.") from error

    def for_task(self, task_type: str) -> list[ModelPlugin]:
        """Return plugins compatible with a project task."""
        return [plugin for plugin in self._plugins.values() if task_type in plugin.supported_tasks]


registry = PluginRegistry()
