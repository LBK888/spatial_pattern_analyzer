"""
Spatial Pattern Analysis GUI Application -- v0.9b 2025-06-09
=======================================

A GUI application for spatial point pattern analysis using Ripley's K and L functions.
Supports CSV and Excel files with flexible column mapping.

Requirements:
- tkinter (usually comes with Python)
- pandas
- numpy
- matplotlib
- scipy
- openpyxl (for Excel support)
- tqdm

Author: GUI Application

To-do: 95% CI analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import queue
from PIL import Image, ImageTk

VERSION_INFO="v0.9b 2025-06-09"

# Import the spatial analyzer (assuming it's in the same directory)
try:
    from ripley_analysis import SpatialPatternAnalyzer, CellData, PatternType, RipleyResult
except ImportError:
    # If the spatial analyzer is not available, create a mock version
    print("Warning: spatial_analyzer module not found. Using mock version.")
    
    class MockSpatialPatternAnalyzer:
        def __init__(self):
            pass
        
        def preprocess_data(self, x, y, cell_types=None):
            return type('CellData', (), {'x': x, 'y': y, 'cell_types': cell_types})()
        
        def analyze_cell_types(self, data, pattern_types=None):
            return {"original": {"Type_1": {"cell_count": len(data.x)}}}
        
        def plot_results(self, results, title=""):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Mock Analysis Results", ha='center', va='center', fontsize=16)
            ax.set_title(title)
            return fig
    
    SpatialPatternAnalyzer = MockSpatialPatternAnalyzer
    
    class PatternType:
        ORIGINAL = "original"
        RANDOM = "random"
        CLUSTER = "cluster"
        REGULAR = "regular"

class DataPreviewDialog:
    """Dialog for previewing and configuring data columns"""
    
    def __init__(self, parent, df: pd.DataFrame):
        self.parent = parent
        self.df = df
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Data Preview and Column Configuration")
        self.dialog.geometry("800x600")
        self.dialog.grab_set()  # Make dialog modal
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI"""
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for data preview
        self.tree = ttk.Treeview(preview_frame)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbars
        tree_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        
        tree_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=tree_scroll_x.set)
        
        # Populate treeview with data
        self.populate_preview()
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Column Configuration")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Header detection
        header_frame = ttk.Frame(config_frame)
        header_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(header_frame, text="Has Header Row:").pack(side=tk.LEFT)
        self.has_header_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(header_frame, variable=self.has_header_var, 
                       command=self.on_header_change).pack(side=tk.LEFT, padx=(5, 0))
        
        # Column selection
        columns = list(self.df.columns)
        
        # X coordinate column
        x_frame = ttk.Frame(config_frame)
        x_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(x_frame, text="X Coordinate Column:", width=20).pack(side=tk.LEFT)
        self.x_column_var = tk.StringVar()
        x_combo = ttk.Combobox(x_frame, textvariable=self.x_column_var, values=columns)
        x_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Y coordinate column
        y_frame = ttk.Frame(config_frame)
        y_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(y_frame, text="Y Coordinate Column:", width=20).pack(side=tk.LEFT)
        self.y_column_var = tk.StringVar()
        y_combo = ttk.Combobox(y_frame, textvariable=self.y_column_var, values=columns)
        y_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Cell type column (optional)
        type_frame = ttk.Frame(config_frame)
        type_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(type_frame, text="Cell Type Column:", width=20).pack(side=tk.LEFT)
        self.type_column_var = tk.StringVar()
        type_values = ["(None)"] + columns
        type_combo = ttk.Combobox(type_frame, textvariable=self.type_column_var, values=type_values)
        type_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Auto-detect likely columns
        self.auto_detect_columns()
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT)
        
    def populate_preview(self):
        """Populate the treeview with data preview"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Setup columns
        columns = list(self.df.columns)
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        # Configure column headings and widths
        for col in columns:
            self.tree.heading(col, text=str(col))
            self.tree.column(col, width=100, minwidth=50)
        
        # Add data rows (limit to first 100 rows for performance)
        for idx, row in self.df.head(100).iterrows():
            values = [str(val) for val in row.values]
            self.tree.insert("", tk.END, values=values)
        
        # Add info about total rows
        if len(self.df) > 100:
            info_values = [f"... ({len(self.df)} total rows)"] + [""] * (len(columns) - 1)
            self.tree.insert("", tk.END, values=info_values)
    
    def auto_detect_columns(self):
        """Automatically detect likely X, Y, and type columns"""
        columns = list(self.df.columns)
        
        # Try to detect X column
        x_candidates = [col for col in columns if 'x' in str(col).lower()]
        if x_candidates:
            self.x_column_var.set(x_candidates[0])
        elif len(columns) >= 2:
            # Look for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                self.x_column_var.set(numeric_cols[0])
        
        # Try to detect Y column
        y_candidates = [col for col in columns if 'y' in str(col).lower()]
        if y_candidates:
            self.y_column_var.set(y_candidates[0])
        elif len(columns) >= 2:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                self.y_column_var.set(numeric_cols[1])
        
        # Try to detect type column
        type_candidates = [col for col in columns if any(keyword in str(col).lower() 
                          for keyword in ['type', 'class', 'category', 'label'])]
        if type_candidates:
            self.type_column_var.set(type_candidates[0])
        else:
            self.type_column_var.set("(None)")
    
    def on_header_change(self):
        """Handle header checkbox change"""
        # This could be extended to reprocess the data if needed
        pass
    
    def ok(self):
        """Handle OK button click"""
        x_col = self.x_column_var.get()
        y_col = self.y_column_var.get()
        type_col = self.type_column_var.get()
        
        if not x_col or not y_col:
            messagebox.showerror("Error", "Please select both X and Y coordinate columns.")
            return
        
        if x_col == y_col:
            messagebox.showerror("Error", "X and Y columns must be different.")
            return
        
        # Prepare result
        self.result = {
            'has_header': self.has_header_var.get(),
            'x_column': x_col,
            'y_column': y_col,
            'type_column': type_col if type_col != "(None)" else None
        }
        
        self.dialog.destroy()
    
    def cancel(self):
        """Handle Cancel button click"""
        self.result = None
        self.dialog.destroy()

class SpatialAnalysisGUI:
    """Main GUI application for spatial pattern analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"Spatial Pattern Analysis Tool - {VERSION_INFO}")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.data = None
        self.analyzer = SpatialPatternAnalyzer()
        self.results = None
        self.current_file = None
        
        # Create queue for thread communication
        self.queue = queue.Queue()
        
        self.setup_ui()
        self.setup_menu()
        
        # Start checking queue
        self.check_queue()
    
    def setup_menu(self):
        """Setup the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open File...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Analysis", command=self.run_analysis)
        analysis_menu.add_command(label="Clear Results", command=self.clear_results)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.open_file())
    
    def setup_ui(self):
        """Setup the main UI"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel for results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.setup_left_panel(left_frame)
        
        # Progress bar (放在左側，所有控制元件下方)
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(left_frame, textvariable=self.progress_var).pack(fill=tk.X, padx=5, pady=(10, 0))
        self.progress_bar = ttk.Progressbar(left_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        self.setup_right_panel(right_frame)
    
    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # File section
        file_frame = ttk.LabelFrame(parent, text="Data File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        file_button_frame = ttk.Frame(file_frame)
        file_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_button_frame, text="Open File", 
                  command=self.open_file).pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(file_button_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Data info section
        info_frame = ttk.LabelFrame(parent, text="Data Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Analysis options section
        options_frame = ttk.LabelFrame(parent, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Pattern types selection
        ttk.Label(options_frame, text="Pattern Types to Analyze:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        
        self.pattern_vars = {}
        pattern_types = [
            ("Original", PatternType.ORIGINAL),
            ("Random", PatternType.RANDOM),
            ("Cluster", PatternType.CLUSTER),
            ("Regular", PatternType.REGULAR)
        ]
        
        for name, pattern_type in pattern_types:
            var = tk.BooleanVar(value=(pattern_type == PatternType.ORIGINAL))
            self.pattern_vars[pattern_type] = var
            ttk.Checkbutton(options_frame, text=name, variable=var).pack(anchor=tk.W, padx=20)
        
        # Analysis parameters
        params_frame = ttk.Frame(options_frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Scan Scope:").grid(row=0, column=0, sticky=tk.W)
        self.scan_scope_var = tk.DoubleVar(value=6.0)
        ttk.Entry(params_frame, textvariable=self.scan_scope_var, width=10).grid(row=0, column=1, padx=(5, 0))
        
        ttk.Label(params_frame, text="Shrink Factor:").grid(row=1, column=0, sticky=tk.W)
        self.shrink_factor_var = tk.DoubleVar(value=0.8)
        ttk.Entry(params_frame, textvariable=self.shrink_factor_var, width=10).grid(row=1, column=1, padx=(5, 0))
        
        ttk.Label(params_frame, text="Number of Radii:").grid(row=2, column=0, sticky=tk.W)
        self.n_radii_var = tk.IntVar(value=100)
        ttk.Entry(params_frame, textvariable=self.n_radii_var, width=10).grid(row=2, column=1, padx=(5, 0))
        
        # Run analysis button
        self.run_button = ttk.Button(options_frame, text="Run Analysis", 
                                   command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(pady=10)

        # Add export buttons frame
        export_frame = ttk.Frame(parent)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(export_frame, text="Export Figure", 
                  command=self.export_figure).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Results", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
    
    def setup_right_panel(self, parent):
        """Setup the right results panel"""
        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        

        
        # Placeholder tab
        placeholder_frame = ttk.Frame(self.notebook)
        self.notebook.add(placeholder_frame, text="Welcome")
        
        welcome_text = """
Welcome to Spatial Pattern Analysis Tool!

Instructions:
1. Click 'Open File' to load your data (CSV or Excel format)
2. Configure column mapping in the preview dialog
3. Select analysis options
4. Click 'Run Analysis' to start

Supported file formats:
- CSV files (.csv)
- Excel files (.xlsx, .xls)

Data format should include:
- X coordinate column
- Y coordinate column  
- Optional: Cell type/category column
        """
        
        ttk.Label(placeholder_frame, text=welcome_text, justify=tk.LEFT).pack(padx=20, pady=20)
    
    def open_file(self):
        """Open and load a data file"""
        filetypes = [
            ("All Supported", "*.csv;*.xlsx;*.xls"), 
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=filetypes
        )
        
        if not filename:
            return
        
        try:
            # Load file based on extension
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(filename)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filename)
            else:
                messagebox.showerror("Error", f"Unsupported file format: {file_ext}")
                return
            
            # Show preview dialog
            preview_dialog = DataPreviewDialog(self.root, df)
            self.root.wait_window(preview_dialog.dialog)
            
            if preview_dialog.result is None:
                return  # User cancelled
            
            config = preview_dialog.result
            
            # Process the data based on configuration
            if not config['has_header']:
                # If no header, use numeric column names
                df.columns = [f"Column_{i}" for i in range(len(df.columns))]
            
            # Extract coordinates and cell types
            x_data = df[config['x_column']].values
            y_data = df[config['y_column']].values
            
            if config['type_column']:
                type_data = df[config['type_column']].values
            else:
                type_data = np.array(['All'] * len(x_data))
            
            # Create CellData object
            self.data = self.analyzer.preprocess_data(x_data, y_data, type_data)
            
            # Update UI
            self.current_file = filename
            self.file_label.config(text=Path(filename).name)
            self.run_button.config(state=tk.NORMAL)
            
            # Update info text
            self.update_info_display()
            
            messagebox.showinfo("Success", f"Data loaded successfully!\n"
                              f"Total points: {len(self.data.x)}\n"
                              f"Cell types: {len(np.unique(self.data.cell_types))}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def update_info_display(self):
        """Update the information display"""
        if self.data is None:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "No data loaded.")
            return
        
        info_text = f"File: {Path(self.current_file).name}\n"
        info_text += f"Total points: {len(self.data.x)}\n"
        info_text += f"X range: {np.min(self.data.x):.2f} - {np.max(self.data.x):.2f}\n"
        info_text += f"Y range: {np.min(self.data.y):.2f} - {np.max(self.data.y):.2f}\n"
        
        if self.data.cell_types is not None:
            unique_types, counts = np.unique(self.data.cell_types, return_counts=True)
            info_text += f"\nCell types ({len(unique_types)}):\n"
            for cell_type, count in zip(unique_types, counts):
                info_text += f"  {cell_type}: {count}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)
    
    def run_analysis(self):
        """Run the spatial analysis in a separate thread"""
        if self.data is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        # Get selected pattern types
        selected_patterns = [pattern_type for pattern_type, var in self.pattern_vars.items() 
                           if var.get()]
        
        if not selected_patterns:
            messagebox.showerror("Error", "Please select at least one pattern type.")
            return
        
        # Update analyzer parameters
        self.analyzer.scan_scope = self.scan_scope_var.get()
        self.analyzer.shrink_factor = self.shrink_factor_var.get()
        self.analyzer.n_radii = self.n_radii_var.get()
        
        # Start analysis in separate thread
        self.progress_var.set("Running analysis...")
        self.progress_bar.start()
        self.run_button.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self._run_analysis_thread, args=(selected_patterns,))
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self, selected_patterns):
        """Run analysis in background thread"""
        try:
            self.queue.put(('progress', f"Calculating Ripley Function..."))
            results = self.analyzer.analyze_cell_types(self.data, pattern_types=selected_patterns)
            self.queue.put(('success', results))
        except Exception as e:
            self.queue.put(('error', str(e)))
    
    def check_queue(self):
        """Check for messages from background thread"""
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'success':
                    self.results = data
                    self.display_results()
                    self.progress_var.set("Analysis completed")
                    self.progress_bar.stop()
                    self.run_button.config(state=tk.NORMAL)
                    
                elif msg_type == 'error':
                    messagebox.showerror("Analysis Error", f"Analysis failed:\n{data}")
                    self.progress_var.set("Analysis failed")
                    self.progress_bar.stop()
                    self.run_button.config(state=tk.NORMAL)
                    
                elif msg_type == 'progress':
                    self.progress_var.set(data)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def display_results(self):
        """Display analysis results in the notebook"""
        if self.results is None:
            return
        
        # Clear existing result tabs
        for tab_id in self.notebook.tabs()[1:]:  # Keep welcome tab
            self.notebook.forget(tab_id)
        
        # Create a single figure for all patterns
        tab_frame = None
        try:
            # 創建高解析度圖形
            fig = self.analyzer.plot_results(self.results, "Analysis Results")
            
            # 將圖形渲染成高解析度圖片
            #fig.set_dpi(300)
            #fig.set_size_inches(18, 12)
            
            # 將 matplotlib 圖形轉換為 PIL Image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            pil_image = Image.frombytes('RGBA', canvas.get_width_height(), buf)
            
            # Create a single tab for all results
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text="Analysis Results")
            
            # 創建一個框架來容納畫布和工具欄
            canvas_frame = ttk.Frame(tab_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            # 創建 tkinter Canvas 來顯示圖片
            self.result_canvas = tk.Canvas(canvas_frame, bg='white')
            self.result_canvas.pack(fill=tk.BOTH, expand=True)
            
            # 將 PIL Image 轉換為 PhotoImage
            self.original_image = pil_image
            self.zoom_factor = 0.15  # 初始縮放比例為15%
            
            # 調整圖片大小
            new_width = int(pil_image.width * self.zoom_factor)
            new_height = int(pil_image.height * self.zoom_factor)
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.result_image = ImageTk.PhotoImage(resized_image)
            
            # 在 Canvas 上顯示圖片
            self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.result_image)
            
            # 添加縮放控制
            control_frame = ttk.Frame(tab_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # 縮放控制
            zoom_frame = ttk.LabelFrame(control_frame, text="Zoom Control")
            zoom_frame.pack(side=tk.LEFT, padx=5)
            
            zoom_in_btn = ttk.Button(zoom_frame, text="Zoom In", width=8,
                                   command=lambda: self._zoom_canvas(1.2))
            zoom_in_btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            zoom_out_btn = ttk.Button(zoom_frame, text="Zoom Out", width=8,
                                    command=lambda: self._zoom_canvas(0.8))
            zoom_out_btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            reset_btn = ttk.Button(zoom_frame, text="Reset View", width=8,
                                 command=self._reset_canvas)
            reset_btn.pack(side=tk.LEFT, padx=2, pady=2)
            
            # 添加滾動條
            h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, 
                                   command=self.result_canvas.xview)
            h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            
            v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                                   command=self.result_canvas.yview)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            self.result_canvas.configure(xscrollcommand=h_scroll.set,
                                      yscrollcommand=v_scroll.set)
            
            # 設置 Canvas 的滾動區域
            self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
            
            # 添加滑鼠滾輪縮放
            self.result_canvas.bind("<MouseWheel>", self._on_mousewheel)
            
            # 添加拖曳功能
            self.result_canvas.bind("<ButtonPress-1>", self._start_pan)
            self.result_canvas.bind("<B1-Motion>", self._pan)
            
            # 初始化拖曳變數
            self.pan_start_x = 0
            self.pan_start_y = 0
            
            # Close the figure to prevent memory leaks
            plt.close(fig)
            
        except Exception as e:
            # Fallback to text display if plotting fails
            if tab_frame is None:
                tab_frame = ttk.Frame(self.notebook)
                self.notebook.add(tab_frame, text="Analysis Results")
            error_label = ttk.Label(tab_frame, text=f"Error creating plot: {str(e)}")
            error_label.pack(expand=True)
        
        # Switch to results tab
        if len(self.notebook.tabs()) > 1:
            self.notebook.select(1)
    
    def _zoom_canvas(self, factor):
        """縮放畫布內容"""
        self.zoom_factor *= factor
        
        # 獲取當前可見區域的中心點
        bbox = self.result_canvas.bbox("all")
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # 重新調整圖片大小
        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)
        
        # 使用 PIL 的 resize 方法重新調整圖片大小
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.result_image = ImageTk.PhotoImage(resized_image)
        
        # 更新畫布
        self.result_canvas.delete("all")
        self.result_canvas.create_image(center_x, center_y, anchor=tk.CENTER, image=self.result_image)
        
        # 更新滾動區域
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
    
    def _reset_canvas(self):
        """重置畫布視圖"""
        self.zoom_factor = 0.25  # 重置為25%
        
        # 重新調整圖片大小
        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)
        resized_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.result_image = ImageTk.PhotoImage(resized_image)
        
        # 更新畫布
        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.result_image)
        self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        """處理滑鼠滾輪事件"""
        if event.delta > 0:
            self._zoom_canvas(1.1)
        else:
            self._zoom_canvas(0.9)
    
    def _start_pan(self, event):
        """開始拖曳"""
        self.result_canvas.scan_mark(event.x, event.y)
    
    def _pan(self, event):
        """拖曳畫布"""
        self.result_canvas.scan_dragto(event.x, event.y, gain=1)
    
    def clear_results(self):
        """Clear analysis results"""
        self.results = None
        
        # Remove result tabs
        for tab_id in self.notebook.tabs()[1:]:  # Keep welcome tab
            self.notebook.forget(tab_id)
        
        # Switch to welcome tab
        self.notebook.select(0)
        
        self.progress_var.set("Results cleared")
    
    def export_results(self):
        """Export analysis results"""
        if self.results is None:
            messagebox.showerror("Error", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create and save the plot
                fig = self.analyzer.plot_results(self.results, "Exported Analysis Results")
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results:\n{str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Spatial Pattern Analysis Tool
Version 1.0

A GUI application for analyzing spatial point patterns using Ripley's K and L functions.

Features:
- Support for CSV and Excel files
- Flexible column mapping
- Multiple pattern analysis
- Interactive visualization

Created using Python, tkinter, and matplotlib.
        """
        messagebox.showinfo("About", about_text)

    def export_figure(self):
        """Export current figure as image file"""
        if self.results is None:
            messagebox.showerror("錯誤", "沒有可用的分析結果。")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            title="儲存圖片"
        )
        
        if filename:
            try:
                # 創建高解析度圖形
                fig = self.analyzer.plot_results(self.results, "Analysis Results")
                fig.set_dpi(300)
                
                # 保存圖片
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                messagebox.showinfo("成功", f"圖片已儲存至：\n{filename}")
            except Exception as e:
                messagebox.showerror("錯誤", f"儲存圖片時發生錯誤：\n{str(e)}")
                if 'fig' in locals():
                    plt.close(fig)

    def export_data(self):
        """Export analysis data as CSV file"""
        if self.results is None:
            messagebox.showerror("錯誤", "沒有可用的分析結果。")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ],
            title="儲存分析數據"
        )
        
        if filename:
            try:
                # Prepare data for export
                export_data = []
                
                for pattern_name, pattern_data in self.results.items():
                    for cell_type, type_data in pattern_data.items():
                        # Get Ripley results
                        ripley_result = type_data['ripley_result']
                        local_analysis = type_data['local_analysis']
                        
                        # Create data rows
                        for i, (radius, k_value, l_value) in enumerate(zip(
                            ripley_result.radii,
                            ripley_result.k_values,
                            ripley_result.l_values
                        )):
                            export_data.append({
                                'Pattern': pattern_name,
                                'Cell_Type': cell_type,
                                'Radius': radius,
                                'K_Value': k_value,
                                'L_Value': l_value,
                                'Cell_Count': type_data['cell_count']
                            })
                
                # Convert to DataFrame
                df = pd.DataFrame(export_data)
                
                # Save based on file extension
                file_ext = Path(filename).suffix.lower()
                if file_ext == '.csv':
                    df.to_csv(filename, index=False)
                elif file_ext == '.xlsx':
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False)
                
                messagebox.showinfo("成功", f"分析數據已儲存至：\n{filename}")
            except Exception as e:
                messagebox.showerror("錯誤", f"儲存數據時發生錯誤：\n{str(e)}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = SpatialAnalysisGUI(root)
    
    # Set minimum window size
    root.minsize(800, 600)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()
