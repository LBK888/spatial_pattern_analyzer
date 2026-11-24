"""Apply the system color scheme consistently to custom and native Qt controls."""

from pathlib import Path
from string import Template

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette


LIGHT = dict(
    window="#f3f6f8", surface="#ffffff", ink="#172c3c", muted="#526779",
    disabled="#657989", divider="#c5d3dd", soft="#eaf0f5", hover="#e7f1fa",
    select="#d9ecfb", link="#125e9b", primary="#1467a7", primary_hover="#10558a",
    banner="#eaf3fb", focus="#207acc", shadow="#a9bdcd", control_border="#647d91",
)
DARK = dict(
    window="#151d25", surface="#202c37", ink="#edf4fa", muted="#b4c6d6",
    disabled="#99acbd", divider="#52677a", soft="#293846", hover="#31475b",
    select="#304e69", link="#9accfa", primary="#1767a4", primary_hover="#1c76b9",
    banner="#243d53", focus="#83bff4", shadow="#111820", control_border="#90aabd",
)

STYLE = Template("""
QMainWindow, QDialog { background: $window; }
QScrollArea { background: $window; }
QFrame#CardFrame, QWidget#CardFrame {
    background: $surface; border: 1px solid $divider; border-radius: 8px;
}
QLabel#TitleLabel { font-size: 20px; font-weight: 700; }
QLabel#HeroTitle { font-size: 23px; font-weight: 700; }
QLabel#HeroBody { font-size: 13px; color: $muted; }
QLabel#SectionLabel { font-weight: 700; }
QLabel#MutedLabel { color: $muted; }
QLabel#BannerLabel {
    background: $banner; border: 1px solid $divider; border-radius: 6px;
    padding: 6px 10px; color: $ink;
}
QGroupBox {
    font-weight: 600; border: 1px solid $divider; border-radius: 6px;
    margin-top: 8px; padding-top: 10px; background: $surface;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton, QLineEdit, QComboBox {
    background: $surface; color: $ink; border: 1px solid $divider; border-radius: 5px;
}
QPushButton { padding: 6px 12px; }
QPushButton:hover { background: $hover; }
QPushButton:focus, QLineEdit:focus, QComboBox:focus { border-color: $focus; }
QPushButton#SmallBtn { padding: 3px 8px; }
QPushButton#PrimaryBtn { background: $primary; color: #ffffff; font-weight: 600; }
QPushButton#PrimaryBtn:hover { background: $primary_hover; }
QPushButton:disabled, QPushButton#PrimaryBtn:disabled, QLineEdit:disabled, QComboBox:disabled {
    background: $soft; color: $disabled; border-color: $divider;
}
QToolButton#HelpBtn {
    background: transparent; border: 1px solid transparent; color: $link;
    font-weight: 700; padding: 0px 3px;
}
QToolButton#HelpBtn:hover, QToolButton#HelpBtn:focus { border-color: $focus; border-radius: 3px; }
QLineEdit, QComboBox { padding: 4px 7px; min-height: 22px; }
QComboBox QAbstractItemView { background: $surface; color: $ink; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid $control_border;
    border-radius: 3px; background: $surface;
}
QCheckBox::indicator:hover, QCheckBox::indicator:focus { border-color: $focus; }
QCheckBox::indicator:disabled { background: $soft; border-color: $control_border; }
QCheckBox::indicator:checked {
    background: $primary; border-color: $link; image: url("$checkmark");
}
QCheckBox::indicator:checked:disabled { background: $muted; border-color: $control_border; }
QTabWidget::pane { border: 1px solid $divider; border-radius: 6px; background: $surface; }
QTabBar::tab {
    background: $soft; border: 1px solid $divider; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    padding: 7px 12px; margin-right: 2px; color: $muted;
}
QTabBar::tab:selected { background: $surface; color: $ink; font-weight: 600; }
QTableWidget {
    background: $surface; alternate-background-color: $window; gridline-color: $divider;
    border: 1px solid $divider; border-radius: 5px; selection-background-color: $select;
}
QHeaderView::section {
    background: $soft; color: $ink; font-weight: 600; padding: 5px 7px;
    border: none; border-right: 1px solid $divider; border-bottom: 1px solid $divider;
}
QProgressBar {
    border: 1px solid $divider; border-radius: 4px; background: $surface;
    text-align: center; min-height: 14px;
}
QProgressBar::chunk { background: $select; border-radius: 3px; }
QToolTip { background: $surface; color: $ink; border: 1px solid $divider; padding: 3px; }
""")


def apply_theme(app, scheme):
    if app.style().objectName().casefold() != "fusion":
        app.setStyle("Fusion")
    colors = DARK if scheme == Qt.ColorScheme.Dark else LIGHT
    roles = QPalette.ColorRole
    groups = QPalette.ColorGroup
    palette = QPalette()
    for role, color in {
        roles.Window: "window", roles.Base: "surface", roles.AlternateBase: "soft",
        roles.WindowText: "ink", roles.Text: "ink", roles.ButtonText: "ink", roles.Button: "surface",
        roles.Highlight: "select", roles.HighlightedText: "ink", roles.Link: "link",
        roles.ToolTipBase: "surface", roles.ToolTipText: "ink", roles.PlaceholderText: "muted",
        roles.Light: "divider", roles.Midlight: "divider", roles.Mid: "divider",
        roles.Dark: "muted", roles.Shadow: "shadow",
    }.items():
        palette.setColor(role, QColor(colors[color]))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(groups.Disabled, role, QColor(colors["disabled"]))
    palette.setColor(groups.Disabled, roles.Base, QColor(colors["soft"]))
    app.setPalette(palette)
    checkmark = Path(__file__).resolve().with_name("ui") / "checked.svg"
    app.setStyleSheet(STYLE.substitute(colors, checkmark=checkmark.as_posix()))
