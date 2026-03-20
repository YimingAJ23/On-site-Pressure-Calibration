# pressure_gui.py

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

from pressure_core import (
    run_pressure_calibration,
    CALIBRANT_DEFAULTS,
    get_material_defaults,
    v0_from_lattice,
)


class PressureGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("On-site Pressure Calibration")
        self.geometry("1254x745")

        self._img_pil = None
        self._imgtk = None
        self._last_png = None

        # track whether user has manually edited EOS params
        self._user_edited_b0 = False
        self._user_edited_b0p = False
        self._user_edited_v0 = False

        self._build_ui()
        self._apply_defaults_for_material(self.material_var.get())

    # ----------------------------
    # UI build
    # ----------------------------
    def _build_ui(self):
        # Root layout: left(control) + right(image)
        root = tk.Frame(self, padx=10, pady=10)
        root.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Make left look consistent
        left.columnconfigure(0, weight=0)
        left.columnconfigure(1, weight=1)

        # ----------------------------
        # Section: Files
        # ----------------------------
        files = ttk.LabelFrame(left, text="Files", padding=(10, 8))
        files.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        files.columnconfigure(1, weight=1)

        self.poni_var = tk.StringVar()
        self.mask_var = tk.StringVar()
        self.data_var = tk.StringVar()
        self.h5_var = tk.StringVar()
        self.mask_none_var = tk.BooleanVar(value=False)

        self._grid_file_row(files, 0, "PONI (.poni)", self.poni_var, browse=True)
        self._grid_file_row(files, 1, "Mask (.tif/.mask)", self.mask_var, browse=True)

        mask_opt = tk.Frame(files)
        mask_opt.grid(row=2, column=1, sticky="w", pady=(2, 0))
        tk.Checkbutton(mask_opt, text="No mask", variable=self.mask_none_var).pack(side=tk.LEFT)

        self._grid_file_row(files, 3, "Data (.edf/.tif/.h5)", self.data_var, browse=True)
        self._grid_file_row(files, 4, "H5 dataset (optional)", self.h5_var, browse=False)

        # ----------------------------
        # Section: Sample / Peak
        # ----------------------------
        sample = ttk.LabelFrame(left, text="Sample / Peak", padding=(10, 8))
        sample.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        sample.columnconfigure(1, weight=1)

        self.material_var = tk.StringVar(value=list(CALIBRANT_DEFAULTS.keys())[0])
        self.hkl_var = tk.StringVar()
        self.crystal_var = tk.StringVar(value="fcc")

        tk.Label(sample, text="Calibrant").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.material_box = ttk.Combobox(
            sample,
            textvariable=self.material_var,
            values=list(CALIBRANT_DEFAULTS.keys()),
            state="readonly",
            width=18
        )
        self.material_box.grid(row=0, column=1, sticky="w", pady=4)
        self.material_box.bind(
            "<<ComboboxSelected>>",
            lambda e: self._apply_defaults_for_material(self.material_var.get())
        )

        tk.Label(sample, text="HKL").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        tk.Entry(sample, textvariable=self.hkl_var, width=10).grid(row=1, column=1, sticky="w", pady=4)

        tk.Label(sample, text="Crystal").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.crystal_box = ttk.Combobox(
            sample,
            textvariable=self.crystal_var,
            values=["bcc", "fcc", "hcp"],
            state="readonly",
            width=10
        )
        self.crystal_box.grid(row=2, column=1, sticky="w", pady=4)
        self.crystal_box.bind("<<ComboboxSelected>>", lambda e: self._on_crystal_changed())

        # ----------------------------
        # Section: Fit window + Run button (same row block)
        # ----------------------------
        fitrun = ttk.LabelFrame(left, text="Fit window (optional)", padding=(10, 8))
        fitrun.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        fitrun.columnconfigure(1, weight=1)
        fitrun.columnconfigure(2, weight=0)

        self.fit_min_var = tk.StringVar(value="")
        self.fit_max_var = tk.StringVar(value="")

        tk.Label(fitrun, text="2θ min").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        tk.Entry(fitrun, textvariable=self.fit_min_var, width=10).grid(row=0, column=1, sticky="w", pady=4)

        tk.Label(fitrun, text="2θ max").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        tk.Entry(fitrun, textvariable=self.fit_max_var, width=10).grid(row=1, column=1, sticky="w", pady=4)

        # Run button to the right, spanning both rows
        self.run_btn = tk.Button(
            fitrun,
            text="Run Calibration",
            command=self.run,
            width=18,
            height=3,
            font=("Arial", 12, "bold")
        )
        self.run_btn.grid(row=0, column=2, rowspan=2, sticky="e", padx=(12, 0), pady=2)

        # ----------------------------
        # Section: EOS
        # ----------------------------
        eos = ttk.LabelFrame(left, text="EOS", padding=(10, 8))
        eos.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        eos.columnconfigure(1, weight=1)

        self.eos_var = tk.StringVar(value="BM3")
        self.b0_var = tk.StringVar()
        self.b0p_var = tk.StringVar()
        self.v0_var = tk.StringVar()

        tk.Label(eos, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        model_row = tk.Frame(eos)
        model_row.grid(row=0, column=1, sticky="w", pady=4)
        tk.Radiobutton(model_row, text="BM3", value="BM3", variable=self.eos_var).pack(side=tk.LEFT)
        tk.Radiobutton(model_row, text="Vinet", value="VINET", variable=self.eos_var).pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(eos, text="B0 (GPa)").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        b0_entry = tk.Entry(eos, textvariable=self.b0_var, width=12)
        b0_entry.grid(row=1, column=1, sticky="w", pady=4)

        tk.Label(eos, text="B0'").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        b0p_entry = tk.Entry(eos, textvariable=self.b0p_var, width=12)
        b0p_entry.grid(row=2, column=1, sticky="w", pady=4)

        tk.Label(eos, text="V0 (Å³)").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        v0_entry = tk.Entry(eos, textvariable=self.v0_var, width=18)
        v0_entry.grid(row=3, column=1, sticky="w", pady=4)

        # track user edits
        b0_entry.bind("<KeyRelease>", lambda e: self._set_user_edited("b0"))
        b0p_entry.bind("<KeyRelease>", lambda e: self._set_user_edited("b0p"))
        v0_entry.bind("<KeyRelease>", lambda e: self._set_user_edited("v0"))

        # ----------------------------
        # Bottom-left: reference label only (no frame / no white box)
        # ----------------------------
        self.ref_label = tk.Label(
            left,
            text="",
            anchor="w",
            justify="left",
            wraplength=420,
            fg="gray25"
        )
        self.ref_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        
        self.copyright_label = tk.Label(
            left,
            text="Yiming Wang, HPSTAR, yiming.wang@hpstar.ac.cn",
            anchor="w",
            justify="left",
            wraplength=420,
            fg="gray25"
        )
        self.copyright_label.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        # Keep left sections anchored to top
        left.rowconfigure(5, weight=1)

        # ----------------------------
        # Right: image preview
        # ----------------------------
        self.image_label = tk.Label(
            right,
            text="Fit preview will appear here after running calibration.",
            anchor="center",
            justify="center"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right.bind("<Configure>", self._on_right_resize)
        self._right_frame = right

    def _grid_file_row(self, parent, row, label, var, browse: bool):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ent = tk.Entry(parent, textvariable=var, width=40)
        ent.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            tk.Button(parent, text="Browse", command=lambda: self._browse_file(var)).grid(
                row=row, column=2, sticky="e", padx=(8, 0), pady=4
            )

    # ----------------------------
    # Helpers
    # ----------------------------
    def _browse_file(self, var):
        path = filedialog.askopenfilename()
        if path:
            var.set(path)

    @staticmethod
    def _to_float_or_none(s: str):
        s = (s or "").strip()
        return None if s == "" else float(s)

    def _set_user_edited(self, which: str):
        if which == "b0":
            self._user_edited_b0 = True
        elif which == "b0p":
            self._user_edited_b0p = True
        elif which == "v0":
            self._user_edited_v0 = True

    # ----------------------------
    # Defaults application
    # ----------------------------
    def _apply_defaults_for_material(self, material: str):
        try:
            cfg = get_material_defaults(material)

            # defaults for HKL & crystal
            self.hkl_var.set(cfg.get("hkl_default", "111"))
            self.crystal_var.set(cfg.get("crystal_default", "fcc"))

            # EOS model default
            self.eos_var.set(cfg.get("eos_default", "BM3"))

            # EOS params defaults (overwrite only if user hasn't edited yet)
            if not self._user_edited_b0:
                self.b0_var.set(str(cfg.get("k0", "")))
            if not self._user_edited_b0p:
                self.b0p_var.set(str(cfg.get("k0p", "")))
            if not self._user_edited_v0:
                self.v0_var.set(f"{float(cfg.get('v0_default', 0.0)):.4f}")

            # reference
            self.ref_label.config(text=str(cfg.get("reference", "")))

            # reset edit flags after applying defaults
            self._user_edited_b0 = False
            self._user_edited_b0p = False
            self._user_edited_v0 = False

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_crystal_changed(self):
        # If user hasn't manually edited V0, update V0 to match chosen crystal using same lattice defaults.
        if self._user_edited_v0:
            return
        material = self.material_var.get().strip()
        if not material:
            return
        cfg = CALIBRANT_DEFAULTS.get(material, {})
        a0 = cfg.get("a0", None)
        c0 = cfg.get("c0", None)
        if a0 is None:
            return
        crystal = (self.crystal_var.get() or "").strip().lower()
        try:
            v0 = v0_from_lattice(crystal, a0, c0)
            self.v0_var.set(f"{float(v0):.4f}")
        except Exception:
            pass

    # ----------------------------
    # Run
    # ----------------------------
    def run(self):
        try:
            poni_path = self.poni_var.get().strip()
            data_path = self.data_var.get().strip()
            if not poni_path:
                raise ValueError("Please select a .poni file.")
            if not data_path:
                raise ValueError("Please select a data file.")

            mask_path = None if self.mask_none_var.get() else self.mask_var.get().strip()
            if mask_path == "":
                mask_path = None

            fit_min = self._to_float_or_none(self.fit_min_var.get())
            fit_max = self._to_float_or_none(self.fit_max_var.get())

            material = self.material_var.get().strip()
            hkl_str = self.hkl_var.get().strip()
            crystal = self.crystal_var.get().strip()

            eos_model = (self.eos_var.get() or "BM3").strip()
            b0 = self._to_float_or_none(self.b0_var.get())
            b0p = self._to_float_or_none(self.b0p_var.get())
            v0 = self._to_float_or_none(self.v0_var.get())

            result = run_pressure_calibration(
                poni_path=poni_path,
                data_path=data_path,
                mask_path=mask_path,
                h5_dset=self.h5_var.get().strip() or None,
                material=material,
                hkl_str=hkl_str,
                crystal=crystal,
                fit_min=fit_min,
                fit_max=fit_max,
                eos_model=eos_model,
                b0=b0,
                b0p=b0p,
                v0=v0,
            )

            if not isinstance(result, dict) or "plot_png" not in result:
                raise RuntimeError("Core did not return plot path.")

            # update reference (material-based)
            self.ref_label.config(text=str(result.get("reference", "")))

            self._load_and_show_image(result["plot_png"])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----------------------------
    # Image display
    # ----------------------------
    def _load_and_show_image(self, path: str):
        self._last_png = path
        self._img_pil = Image.open(path)
        self._render_image_to_label()

    def _on_right_resize(self, event):
        if self._img_pil is not None:
            self._render_image_to_label()

    def _render_image_to_label(self):
        if self._img_pil is None:
            return

        w = max(self._right_frame.winfo_width() - 10, 200)
        h = max(self._right_frame.winfo_height() - 10, 200)

        img = self._img_pil.copy()
        img.thumbnail((w, h))

        self._imgtk = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._imgtk, text="")


if __name__ == "__main__":
    PressureGUI().mainloop()
