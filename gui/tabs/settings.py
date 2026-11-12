"""
Comprehensive Settings Tab with model selection, connection settings, and preferences.
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFileDialog
)
from PySide6.QtCore import Qt, Signal

from qfluentwidgets import (
    ScrollArea, ExpandLayout, SettingCardGroup, PushSettingCard, FluentIcon as FIF,
    setTheme, Theme, PrimaryPushSettingCard, ComboBox, LineEdit, PushButton, 
    InfoBar, InfoBarPosition, SettingCard, Slider, 
    StrongBodyLabel, SwitchButton
)

from core.settings_store import settings


class ComboBoxCard(SettingCard):
    """Setting card with a ComboBox for selection."""
    
    value_changed = Signal(str)
    
    def __init__(self, icon, title, description, options: list, key_path: str, parent=None):
        super().__init__(icon, title, description, parent)
        self.key_path = key_path
        
        self.combo = ComboBox(self)
        self.combo.setMinimumWidth(180)
        self.combo.addItems(options)
        
        # Set current value from settings
        current = settings.get(key_path, options[0] if options else "")
        if current in options:
            self.combo.setCurrentText(current)
        
        self.combo.currentTextChanged.connect(self._on_changed)
        self.hBoxLayout.addWidget(self.combo, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
    
    def _on_changed(self, text: str):
        settings.set(self.key_path, text)
        self.value_changed.emit(text)


class FilePathCard(SettingCard):
    """Setting card for selecting a local model or projector file."""

    value_changed = Signal(str)

    def __init__(self, icon, title, description, key_path: str,
                 file_filter: str, parent=None):
        super().__init__(icon, title, description, parent)
        self.key_path = key_path
        self.file_filter = file_filter

        self.path_input = LineEdit(self)
        self.path_input.setMinimumWidth(320)
        self.path_input.setPlaceholderText("Select a local GGUF file")
        self.path_input.setText(str(settings.get(key_path, "")))
        self.path_input.textChanged.connect(self._on_changed)

        self.browse_button = PushButton("Browse", self)
        self.browse_button.clicked.connect(self._browse)

        self.hBoxLayout.addWidget(self.path_input, 1, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)
        self.hBoxLayout.addWidget(self.browse_button, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)

    def _on_changed(self, value: str):
        settings.set(self.key_path, value.strip())
        self.value_changed.emit(value.strip())

    def _browse(self):
        current = self.path_input.text().strip()
        selected, _ = QFileDialog.getOpenFileName(
            self.window(),
            "Select model file",
            current or str(Path.home()),
            self.file_filter,
        )
        if selected:
            self.path_input.setText(selected)


class SliderCard(SettingCard):
    """Setting card with a slider and value label."""
    
    value_changed = Signal(int)
    
    def __init__(self, icon, title, description, key_path: str, 
                 min_val: int, max_val: int, parent=None):
        super().__init__(icon, title, description, parent)
        self.key_path = key_path
        
        self.value_label = StrongBodyLabel(self)
        self.value_label.setMinimumWidth(30)
        
        self.slider = Slider(Qt.Horizontal, self)
        self.slider.setMinimumWidth(150)
        self.slider.setRange(min_val, max_val)
        
        current = settings.get(key_path, min_val)
        self.slider.setValue(current)
        self.value_label.setText(str(current))
        
        self.slider.valueChanged.connect(self._on_changed)
        
        self.hBoxLayout.addWidget(self.value_label, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)
        self.hBoxLayout.addWidget(self.slider, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
    
    def _on_changed(self, value: int):
        self.value_label.setText(str(value))
        settings.set(self.key_path, value)
        self.value_changed.emit(value)


class SwitchCard(SettingCard):
    """Setting card with a switch toggle."""
    
    checked_changed = Signal(bool)
    
    def __init__(self, icon, title, description, key_path: str, parent=None):
        super().__init__(icon, title, description, parent)
        self.key_path = key_path
        
        self.switch = SwitchButton(self)
        self.switch.setChecked(settings.get(key_path, False))
        self.switch.checkedChanged.connect(self._on_changed)
        
        self.hBoxLayout.addWidget(self.switch, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
    
    def _on_changed(self, checked: bool):
        settings.set(self.key_path, checked)
        self.checked_changed.emit(checked)


class TextInputCard(SettingCard):
    """Setting card with a text input field."""
    
    value_changed = Signal(str)
    
    def __init__(self, icon, title, description, key_path: str, 
                 placeholder: str = "", parent=None):
        super().__init__(icon, title, description, parent)
        self.key_path = key_path
        
        self.input = LineEdit(self)
        self.input.setMinimumWidth(200)
        self.input.setPlaceholderText(placeholder)
        
        current = settings.get(key_path, "")
        self.input.setText(str(current))
        
        self.input.textChanged.connect(self._on_changed)
        
        self.hBoxLayout.addWidget(self.input, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(16)
    
    def _on_changed(self, text: str):
        # Try to convert to float if it looks like a number
        try:
            value = float(text) if text else text
            settings.set(self.key_path, value)
            self.value_changed.emit(text)
        except ValueError:
            # Keep as string
            settings.set(self.key_path, text)
            self.value_changed.emit(text)


class SettingsTab(ScrollArea):
    """Settings for embedded models and optional application capabilities."""

    voice_enabled_changed = Signal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsInterface")
        self.scrollWidget = QWidget()
        self.expandLayout = ExpandLayout(self.scrollWidget)
        
        self.setStyleSheet("background-color: transparent;")
        self.scrollWidget.setObjectName("scrollWidget")
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        
        
        self._init_ui()

    def _init_ui(self):
        # ─────────────────────────────────────────────────────────────
        # Personalization Group
        # ─────────────────────────────────────────────────────────────
        self.personal_group = SettingCardGroup("Personalization", self.scrollWidget)
        
        self.theme_card = ComboBoxCard(
            FIF.BRUSH,
            "Application Theme",
            "Change the appearance of the application",
            ["Light", "Dark", "Auto"],
            "theme",
            self.personal_group
        )
        self.theme_card.value_changed.connect(self._on_theme_changed)
        self.personal_group.addSettingCard(self.theme_card)
        self.expandLayout.addWidget(self.personal_group)

        # ─────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────
        # Unified Local Models (embedded llama.cpp)
        # ─────────────────────────────────────────────────────────────
        self.ai_group = SettingCardGroup("Local AI Models", self.scrollWidget)

        gguf_filter = "GGUF model (*.gguf);;All files (*.*)"
        self.chat_model_path_card = FilePathCard(
            FIF.CHAT,
            "Primary Chat Model — K2-Horizon",
            "Local K2-Horizon GGUF used for coding and normal conversation",
            "models.chat_path",
            gguf_filter,
            self.ai_group
        )
        self.ai_group.addSettingCard(self.chat_model_path_card)

        self.vision_model_path_card = FilePathCard(
            FIF.GLOBE,
            "Vision Model — Gemma 4",
            "Local Gemma 4 GGUF loaded only when an image needs analysis",
            "models.vision_path",
            gguf_filter,
            self.ai_group
        )
        self.ai_group.addSettingCard(self.vision_model_path_card)

        self.vision_projector_path_card = FilePathCard(
            FIF.ROBOT,
            "Gemma 4 Vision Projector",
            "The matching mmproj GGUF file from the same Gemma 4 model release",
            "models.vision_projector_path",
            gguf_filter,
            self.ai_group
        )
        self.ai_group.addSettingCard(self.vision_projector_path_card)

        self.chat_context_card = TextInputCard(
            FIF.HISTORY,
            "K2 Context Length",
            "Maximum context tokens for the primary coding model",
            "models.chat_context",
            "8192",
            self.ai_group
        )
        self.ai_group.addSettingCard(self.chat_context_card)

        self.vision_context_card = TextInputCard(
            FIF.HISTORY,
            "Gemma Context Length",
            "Maximum context tokens for image analysis",
            "models.vision_context",
            "8192",
            self.ai_group
        )
        self.ai_group.addSettingCard(self.vision_context_card)

        self.gpu_layers_card = TextInputCard(
            FIF.INFO,
            "GPU Layers",
            "Use -1 for automatic full offload, or specify a lower layer count",
            "inference.gpu_layers",
            "-1",
            self.ai_group
        )
        self.ai_group.addSettingCard(self.gpu_layers_card)

        self.cpu_threads_card = TextInputCard(
            FIF.INFO,
            "CPU Threads",
            "CPU threads used by local inference",
            "inference.cpu_threads",
            "6",
            self.ai_group
        )
        self.ai_group.addSettingCard(self.cpu_threads_card)

        self.expandLayout.addWidget(self.ai_group)

        # Voice & Audio Group
        # ─────────────────────────────────────────────────────────────
        self.voice_group = SettingCardGroup("Voice & Audio", self.scrollWidget)
        
        piper_voices = [
            "en_GB-alba-medium",
            "en_US-amy-medium",
            "en_US-lessac-medium",
            "en_US-libritts-high",
        ]
        self.tts_voice_card = ComboBoxCard(
            FIF.VOLUME,
            "TTS Voice",
            "Voice model for text-to-speech",
            piper_voices,
            "tts.voice",
            self.voice_group
        )
        self.voice_group.addSettingCard(self.tts_voice_card)

        self.voice_assistant_card = SwitchCard(
            FIF.VOLUME,
            "Voice Assistant",
            "Load speech recognition only when enabled; uses the embedded K2 model",
            "voice.enabled",
            self.voice_group
        )
        self.voice_assistant_card.checked_changed.connect(self.voice_enabled_changed.emit)
        self.voice_group.addSettingCard(self.voice_assistant_card)

        self.expandLayout.addWidget(self.voice_group)

        # ─────────────────────────────────────────────────────────────
        # Weather Location Group
        # ─────────────────────────────────────────────────────────────
        self.weather_group = SettingCardGroup("Weather Location", self.scrollWidget)
        
        self.city_card = TextInputCard(
            FIF.PIN,
            "City Name",
            "Display name for your location",
            "weather.city",
            "New York, NY",
            self.weather_group
        )
        self.weather_group.addSettingCard(self.city_card)
        
        self.latitude_card = TextInputCard(
            FIF.PIN,
            "Latitude",
            "Latitude coordinate (-90 to 90)",
            "weather.latitude",
            "40.7128",
            self.weather_group
        )
        self.weather_group.addSettingCard(self.latitude_card)
        
        self.longitude_card = TextInputCard(
            FIF.GLOBE,
            "Longitude",
            "Longitude coordinate (-180 to 180)",
            "weather.longitude",
            "-74.0060",
            self.weather_group
        )
        self.weather_group.addSettingCard(self.longitude_card)
        
        self.expandLayout.addWidget(self.weather_group)

        # ─────────────────────────────────────────────────────────────
        # General Group
        # ─────────────────────────────────────────────────────────────
        self.general_group = SettingCardGroup("General", self.scrollWidget)
        
        self.max_history_card = SliderCard(
            FIF.HISTORY,
            "Max Chat History",
            "Number of messages to keep in context",
            "general.max_history",
            5, 50,
            self.general_group
        )
        self.general_group.addSettingCard(self.max_history_card)
        
        self.auto_news_card = SwitchCard(
            FIF.DOCUMENT,
            "Auto-fetch News",
            "Automatically fetch news on startup",
            "general.auto_fetch_news",
            self.general_group
        )
        self.general_group.addSettingCard(self.auto_news_card)
        
        self.expandLayout.addWidget(self.general_group)

        # ─────────────────────────────────────────────────────────────
        # About Group
        # ─────────────────────────────────────────────────────────────
        self.about_group = SettingCardGroup("About", self.scrollWidget)
        
        self.about_card = PrimaryPushSettingCard(
            "Check Update",
            FIF.INFO,
            "About A.D.A",
            "Version 0.2.0 (Alpha)",
            self.about_group
        )
        self.about_group.addSettingCard(self.about_card)
        
        self.reset_card = PushSettingCard(
            "Reset",
            FIF.CANCEL,
            "Reset Settings",
            "Restore all settings to defaults",
            self.about_group
        )
        self.reset_card.clicked.connect(self._on_reset)
        self.about_group.addSettingCard(self.reset_card)
        
        self.expandLayout.addWidget(self.about_group)

    def _on_theme_changed(self, value: str):
        theme_map = {"Dark": Theme.DARK, "Light": Theme.LIGHT, "Auto": Theme.AUTO}
        setTheme(theme_map.get(value, Theme.DARK))
    
    def _on_reset(self):
        settings.reset_to_defaults()
        InfoBar.success(
            title="Settings Reset",
            content="All settings restored to defaults. Please restart the app.",
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=5000,
            parent=self.window()
        )
    
