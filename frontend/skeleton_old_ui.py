import os
import sys
import threading
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import backend modules
from modules.blur import blur_images
from modules.metadata import clean_metadata
from modules.compress import compress_and_package
from meta.modules.metadata_tools import read_exif, remove_exif
from blurring.modules.blur_tools import detect_faces, detect_plates, _blur_regions, _blur_text_regions, _tesseract_available
from privacy_score.privacy_score import analyze_image_bytes

# Output directory
OUTPUT_ROOT = ROOT_DIR / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

# Frontend directory for images
FRONTEND_DIR = Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MetaClean - Image Sanitization Tool")
        self.attributes('-fullscreen', True)
        self.bind("<Escape>", lambda e: self.destroy())

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # State management
        self.selected_files: List[str] = []
        self.current_image_index = 0
        self.processed_images: List[str] = []  # Store processed image paths
        self.temp_work_dir = OUTPUT_ROOT / "temp_work"
        self.temp_work_dir.mkdir(exist_ok=True, parents=True)
        self.privacy_meter_cache = {}
        self.current_displayed_image_path: Optional[str] = None

        self.show_home_page()

    def show_home_page(self):
        """Initial home page with Get Started button"""
        self.current_displayed_image_path = None
        for widget in self.winfo_children():
            widget.destroy()

        bg_path = FRONTEND_DIR / "bg.png"
        if not bg_path.exists():
            # Fallback if bg.png doesn't exist
            self.configure(bg="#000000")
            canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, bg="#000000", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
        else:
            bg_image = Image.open(bg_path).resize((self.screen_width, self.screen_height), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Get Started button
        getstarted_path = FRONTEND_DIR / "getstarted.png"
        if getstarted_path.exists():
            getstarted_img = Image.open(getstarted_path)
            scale_factor = 0.7
            getstarted_img = getstarted_img.resize(
                (int(getstarted_img.width * scale_factor), int(getstarted_img.height * scale_factor)),
                Image.LANCZOS
            )
            self.getstarted_photo = ImageTk.PhotoImage(getstarted_img)
            getstarted_btn = tk.Button(
                self,
                image=self.getstarted_photo,
                bd=0,
                highlightthickness=0,
                relief="flat",
                bg="#000000",
                activebackground="#000000",
                command=self.show_upload_page
            )
            canvas.create_window(
                self.screen_width // 2,
                (self.screen_height // 2) + 300,
                anchor="center",
                window=getstarted_btn
            )
        else:
            # Fallback button
            getstarted_btn = tk.Button(
                self,
                text="Get Started",
                font=("Arial", 24, "bold"),
                bg="#4CAF50",
                fg="white",
                padx=40,
                pady=20,
                command=self.show_upload_page
            )
            canvas.create_window(
                self.screen_width // 2,
                (self.screen_height // 2) + 300,
                anchor="center",
                window=getstarted_btn
            )

        # Close button (X)
        close_button = tk.Button(
            self,
            text="×",
            font=("Arial", 30, "bold"),
            fg="white",
            bg="#000000",
            activebackground="darkred",
            relief="flat",
            bd=0,
            command=self.destroy
        )
        canvas.create_window(self.screen_width - 60, 60, anchor="ne", window=close_button)

    def show_upload_page(self):
        """Media upload page with browse file button"""
        self.current_displayed_image_path = None
        for widget in self.winfo_children():
            widget.destroy()

        # Background
        bg2_path = FRONTEND_DIR / "bg2.png"
        if bg2_path.exists():
            bg2_image = Image.open(bg2_path).resize((self.screen_width, self.screen_height), Image.LANCZOS)
            self.bg2_photo = ImageTk.PhotoImage(bg2_image)
            canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.create_image(0, 0, image=self.bg2_photo, anchor="nw")
        else:
            canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, bg="#f0f0f0", highlightthickness=0)
            canvas.pack(fill="both", expand=True)

        # Title
        title_label = tk.Label(
            self,
            text="Upload Your Images",
            font=("Arial", 32, "bold"),
            bg="#f0f0f0" if not bg2_path.exists() else None
        )
        canvas.create_window(self.screen_width // 2, 200, anchor="center", window=title_label)

        # Browse Files button
        browse_btn = tk.Button(
            self,
            text="Browse Files",
            font=("Arial", 20, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=40,
            pady=20,
            command=self.browse_files
        )
        canvas.create_window(self.screen_width // 2, 400, anchor="center", window=browse_btn)

        # File list display
        self.file_listbox = tk.Listbox(
            self,
            font=("Arial", 12),
            height=10,
            width=60
        )
        canvas.create_window(self.screen_width // 2, 600, anchor="center", window=self.file_listbox)

        # Cancel and Next buttons
        cancel_path = FRONTEND_DIR / "cancel.png"
        next_path = FRONTEND_DIR / "next.png"
        
        if cancel_path.exists() and next_path.exists():
            scale_small = 0.4
            cancel_img = Image.open(cancel_path).resize(
                (int(Image.open(cancel_path).width * scale_small),
                 int(Image.open(cancel_path).height * scale_small)),
                Image.LANCZOS
            )
            next_img = Image.open(next_path).resize(
                (int(Image.open(next_path).width * scale_small),
                 int(Image.open(next_path).height * scale_small)),
                Image.LANCZOS
            )
            self.cancel_photo = ImageTk.PhotoImage(cancel_img)
            self.next_photo = ImageTk.PhotoImage(next_img)
            
            cancel_btn = tk.Button(self, image=self.cancel_photo, bd=0, relief="flat", bg="#ffffff", command=self.show_home_page)
            next_btn = tk.Button(self, image=self.next_photo, bd=0, relief="flat", bg="#ffffff", command=self.go_to_metadata_page)
        else:
            cancel_btn = tk.Button(self, text="Cancel", font=("Arial", 16), bg="#f44336", fg="white", padx=30, pady=10, command=self.show_home_page)
            next_btn = tk.Button(self, text="Next", font=("Arial", 16), bg="#2196F3", fg="white", padx=30, pady=10, command=self.go_to_metadata_page)

        button_y = self.screen_height - 150
        canvas.create_window(self.screen_width - 280, button_y, anchor="nw", window=cancel_btn)
        canvas.create_window(self.screen_width - 160, button_y, anchor="nw", window=next_btn)

        # Close button
        close_btn = tk.Button(self, text="×", font=("Arial", 30, "bold"), fg="black", bg="white",
                              activebackground="darkred", relief="flat", bd=0, command=self.destroy)
        canvas.create_window(self.screen_width - 60, 60, anchor="ne", window=close_btn)

    def browse_files(self):
        """Browse and select image files"""
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")]
        )
        if files:
            self.invalidate_privacy_cache()
            self.selected_files = list(files)
            self.file_listbox.delete(0, tk.END)
            for path in self.selected_files:
                self.file_listbox.insert(tk.END, Path(path).name)

    def go_to_metadata_page(self):
        """Go to metadata cleaning page (first image)"""
        if not self.selected_files:
            messagebox.showinfo("No Images", "Please select at least one image first.")
            return
        
        self.current_image_index = 0
        self.processed_images = []
        self.show_metadata_cleaning_page()

    def show_metadata_cleaning_page(self):
        """Show metadata cleaning page for current image"""
        if self.current_image_index >= len(self.selected_files):
            # All images processed, go to download page
            self.show_download_page()
            return

        for widget in self.winfo_children():
            widget.destroy()

        current_image_path = self.selected_files[self.current_image_index]
        total_images = len(self.selected_files)
        self.current_displayed_image_path = current_image_path

        # Main frame
        main_frame = tk.Frame(self, bg="#f5f5f5")
        main_frame.pack(fill="both", expand=True)

        # Counter label (top left)
        counter_label = tk.Label(
            main_frame,
            text=f"Image {self.current_image_index + 1} of {total_images}",
            font=("Arial", 18, "bold"),
            bg="#f5f5f5"
        )
        counter_label.pack(anchor="nw", padx=20, pady=20)

        # Content frame (image on left, menu on right)
        content_frame = tk.Frame(main_frame, bg="#f5f5f5")
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Left side - Image preview + meter (scrollable)
        image_frame = tk.Frame(content_frame, bg="white", relief="sunken", bd=2)
        image_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

        image_canvas = tk.Canvas(image_frame, bg="white", highlightthickness=0)
        image_canvas.pack(side="left", fill="both", expand=True)

        image_scrollbar = tk.Scrollbar(image_frame, orient="vertical", command=image_canvas.yview)
        image_scrollbar.pack(side="right", fill="y")
        image_canvas.configure(yscrollcommand=image_scrollbar.set)

        image_inner = tk.Frame(image_canvas, bg="white")
        image_canvas.create_window((0, 0), window=image_inner, anchor="nw")

        def _update_scroll_region(event):
            image_canvas.configure(scrollregion=image_canvas.bbox("all"))

        image_inner.bind("<Configure>", _update_scroll_region)

        def _on_mousewheel(event):
            image_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        image_canvas.bind("<Enter>", lambda e: image_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        image_canvas.bind("<Leave>", lambda e: image_canvas.unbind_all("<MouseWheel>"))

        try:
            img = Image.open(current_image_path)
            # Resize to fit screen while maintaining aspect ratio
            max_width = self.screen_width // 2 - 100
            max_height = self.screen_height - 200
            
            img.thumbnail((max_width, max_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            image_label = tk.Label(image_inner, image=photo, bg="white")
            image_label.image = photo  # Keep reference
            image_label.pack(padx=10, pady=10)
            meter_container = tk.Frame(image_inner, bg="white")
            meter_container.pack(fill="x", padx=20, pady=(0, 20))
            self.render_privacy_meter(meter_container, current_image_path)
        except Exception as e:
            error_label = tk.Label(image_inner, text=f"Error loading image:\n{e}", fg="red", font=("Arial", 12), bg="white")
            error_label.pack(padx=10, pady=10)

        # Right side - Menu
        menu_frame = tk.Frame(content_frame, bg="white", relief="sunken", bd=2, width=400)
        menu_frame.pack(side="right", fill="y", padx=(20, 0))
        menu_frame.pack_propagate(False)

        menu_title = tk.Label(
            menu_frame,
            text="Processing Options",
            font=("Arial", 20, "bold"),
            bg="white"
        )
        menu_title.pack(pady=20)

        # Menu buttons
        buttons = [
            ("1. Preview Metadata", self.preview_metadata),
            ("2. Remove All Metadata", self.remove_all_metadata),
            ("3. Detect & Blur Faces/Plates/Text", self.blur_faces_interactive),
            ("4. View Privacy Score", self.view_privacy_score),
            ("5. Rename", self.rename_image),
            ("6. Done", self.image_done)
        ]

        for text, command in buttons:
            btn = tk.Button(
                menu_frame,
                text=text,
                font=("Arial", 14),
                bg="#2196F3",
                fg="white",
                padx=20,
                pady=15,
                width=25,
                command=command
            )
            btn.pack(pady=10, padx=20)

        # Back button (top right)
        back_btn = tk.Button(
            main_frame,
            text="Back",
            font=("Arial", 14),
            bg="#757575",
            fg="white",
            padx=20,
            pady=10,
            command=self.show_upload_page
        )
        back_btn.place(x=self.screen_width - 120, y=20)

    def preview_metadata(self):
        """Option 1: Preview metadata"""
        current_image_path = self.selected_files[self.current_image_index]
        
        metadata_window = tk.Toplevel(self)
        metadata_window.title("Image Metadata")
        metadata_window.geometry("700x500")
        
        text_area = scrolledtext.ScrolledText(metadata_window, wrap=tk.WORD, width=80, height=25)
        text_area.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_area.insert(tk.END, f"File: {Path(current_image_path).name}\n")
        text_area.insert(tk.END, "="*70 + "\n\n")
        
        try:
            exif = read_exif(current_image_path)
            if exif:
                for key, value in exif.items():
                    text_area.insert(tk.END, f"{key}: {value}\n")
            else:
                text_area.insert(tk.END, "No EXIF metadata found.\n")
        except Exception as e:
            text_area.insert(tk.END, f"Error reading metadata: {e}\n")
        
        text_area.config(state="disabled")

    def remove_all_metadata(self):
        """Option 2: Remove all metadata"""
        current_image_path = self.selected_files[self.current_image_index]
        
        try:
            # Save cleaned version to temp work directory
            output_path = self.temp_work_dir / f"cleaned_{Path(current_image_path).name}"
            remove_exif(current_image_path, str(output_path))
            
            # Update the current image to the cleaned version
            self.invalidate_privacy_cache(current_image_path)
            self.selected_files[self.current_image_index] = str(output_path)
            
            messagebox.showinfo("Success", "Metadata removed successfully!")
            # Refresh the image preview
            self.show_metadata_cleaning_page()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove metadata:\n{e}")

    def blur_faces_interactive(self):
        """Option 3: Interactive blur faces and plates"""
        current_image_path = self.selected_files[self.current_image_index]
        
        try:
            # Load image
            image = cv2.imread(current_image_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image.")
                return

            # Detect faces and plates
            faces = detect_faces(image)
            plates = detect_plates(image)

            if not faces and not plates:
                # Auto blur text if no faces/plates
                if _tesseract_available():
                    text_count = _blur_text_regions(image)
                    messagebox.showinfo("Blur Complete", f"Blurred {text_count} text regions.")
                else:
                    messagebox.showinfo("No Detections", "No faces or license plates detected.")
                
                # Save blurred image
                output_path = self.temp_work_dir / f"blurred_{Path(current_image_path).name}"
                cv2.imwrite(str(output_path), image)
                self.invalidate_privacy_cache(current_image_path)
                self.selected_files[self.current_image_index] = str(output_path)
                self.show_metadata_cleaning_page()
                return

            # Create preview image with numbered boxes (work on copy)
            display_img = image.copy()
            
            # Draw faces with numbers
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 255), 3)
                cv2.putText(display_img, f"FACE {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw plates with numbers
            for i, (x, y, w, h) in enumerate(plates):
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(display_img, f"PLATE {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show preview window
            preview_window = tk.Toplevel(self)
            preview_window.title("Select Faces and Plates to Blur")
            preview_window.attributes('-fullscreen', True)
            
            # Convert to PIL for display
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            display_pil = Image.fromarray(display_rgb)
            
            # Resize to fit screen
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            display_pil.thumbnail((screen_w - 100, screen_h - 200), Image.LANCZOS)
            display_photo = ImageTk.PhotoImage(display_pil)
            
            canvas = tk.Canvas(preview_window, width=screen_w, height=screen_h, bg="black")
            canvas.pack(fill="both", expand=True)
            canvas.create_image(screen_w // 2, screen_h // 2 - 100, image=display_photo, anchor="center")
            canvas.image = display_photo

            # Instructions
            info_text = f"Detected {len(faces)} face(s) and {len(plates)} license plate(s)\n"
            info_text += "Enter face numbers to blur (comma-separated, e.g., 0,1,2 or 'all'/'none'):"
            info_label = tk.Label(preview_window, text=info_text, font=("Arial", 14), bg="black", fg="white")
            canvas.create_window(screen_w // 2, screen_h - 300, anchor="center", window=info_label)

            # Input fields
            face_entry = tk.Entry(preview_window, font=("Arial", 14), width=50)
            canvas.create_window(screen_w // 2, screen_h - 250, anchor="center", window=face_entry)

            plate_info_text = "Enter plate numbers to blur (comma-separated, e.g., 0,1 or 'all'/'none'):"
            plate_info_label = tk.Label(preview_window, text=plate_info_text, font=("Arial", 14), bg="black", fg="white")
            canvas.create_window(screen_w // 2, screen_h - 200, anchor="center", window=plate_info_label)

            plate_entry = tk.Entry(preview_window, font=("Arial", 14), width=50)
            canvas.create_window(screen_w // 2, screen_h - 150, anchor="center", window=plate_entry)

            def apply_blur():
                try:
                    # Reload original image to work on fresh copy
                    work_image = cv2.imread(current_image_path)
                    if work_image is None:
                        messagebox.showerror("Error", "Could not reload image.")
                        preview_window.destroy()
                        return

                    # Parse face selection
                    face_choice = face_entry.get().strip().lower()
                    if face_choice == "all":
                        faces_to_blur = list(range(len(faces)))
                    elif face_choice == "none" or not face_choice:
                        faces_to_blur = []
                    else:
                        faces_to_blur = [int(i.strip()) for i in face_choice.split(",") if i.strip().isdigit()]

                    # Parse plate selection
                    plate_choice = plate_entry.get().strip().lower()
                    if plate_choice == "all":
                        plates_to_blur = list(range(len(plates)))
                    elif plate_choice == "none" or not plate_choice:
                        plates_to_blur = []
                    else:
                        plates_to_blur = [int(i.strip()) for i in plate_choice.split(",") if i.strip().isdigit()]

                    # Apply blur to selected faces
                    for i, (x, y, w, h) in enumerate(faces):
                        if i in faces_to_blur:
                            roi = work_image[y:y+h, x:x+w]
                            if roi.size > 0:
                                work_image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)

                    # Apply blur to selected plates
                    for i, (x, y, w, h) in enumerate(plates):
                        if i in plates_to_blur:
                            roi = work_image[y:y+h, x:x+w]
                            if roi.size > 0:
                                work_image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (99, 99), 30)

                    # Blur text if Tesseract available
                    text_count = 0
                    if _tesseract_available():
                        text_count = _blur_text_regions(work_image)

                    # Save blurred image
                    output_path = self.temp_work_dir / f"blurred_{Path(current_image_path).name}"
                    cv2.imwrite(str(output_path), work_image)
                    self.invalidate_privacy_cache(current_image_path)
                    self.selected_files[self.current_image_index] = str(output_path)

                    preview_window.destroy()
                    messagebox.showinfo("Blur Complete", f"Blurred {len(faces_to_blur)} face(s), {len(plates_to_blur)} plate(s), and {text_count} text region(s).")
                    self.show_metadata_cleaning_page()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to apply blur:\n{e}")
                    preview_window.destroy()

            apply_btn = tk.Button(preview_window, text="Apply Blur", font=("Arial", 16, "bold"), 
                                 bg="#4CAF50", fg="white", padx=30, pady=10, command=apply_blur)
            canvas.create_window(screen_w // 2, screen_h - 80, anchor="center", window=apply_btn)

            close_btn = tk.Button(preview_window, text="Cancel", font=("Arial", 14), 
                                 bg="#f44336", fg="white", padx=20, pady=10, command=preview_window.destroy)
            canvas.create_window(screen_w - 100, 50, anchor="ne", window=close_btn)

        except Exception as e:
            messagebox.showerror("Error", f"Blurring failed:\n{e}")

    def view_privacy_score(self):
        """Option 4: View privacy score"""
        current_image_path = self.selected_files[self.current_image_index]
        
        try:
            analysis = self.get_privacy_analysis(current_image_path, force_refresh=True)
            if analysis.get("error"):
                raise ValueError(analysis["error"])
            score = analysis.get("score", "N/A")
            risk = analysis.get("risk_level", "unknown").title()
            safe = "Yes" if analysis.get("safe_to_share") else "No"
            reasons = analysis.get("reasons", [])
            recommendations = analysis.get("recommendations", [])

            score_window = tk.Toplevel(self)
            score_window.title("Privacy Score")
            score_window.geometry("650x500")
            
            text_area = scrolledtext.ScrolledText(score_window, wrap=tk.WORD, width=70, height=22)
            text_area.pack(fill="both", expand=True, padx=10, pady=10)
            
            text_area.insert(tk.END, f"File: {Path(current_image_path).name}\n")
            text_area.insert(tk.END, "="*50 + "\n\n")
            text_area.insert(tk.END, f"Privacy Score: {score}/100\n")
            text_area.insert(tk.END, f"Risk Level: {risk}\n")
            text_area.insert(tk.END, f"Safe To Share Now? {safe}\n\n")
            
            if reasons:
                text_area.insert(tk.END, "Why the score looks this way:\n")
                for reason in reasons:
                    text_area.insert(tk.END, f"  - {reason}\n")
                text_area.insert(tk.END, "\n")
            
            if recommendations:
                text_area.insert(tk.END, "Suggested fixes:\n")
                for rec in recommendations:
                    text_area.insert(tk.END, f"  - {rec}\n")
            else:
                text_area.insert(tk.END, "No further privacy risks detected.\n")
            
            text_area.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute privacy score:\n{e}")

    def render_privacy_meter(self, parent: tk.Frame, image_path: str):
        """Create and populate the privacy meter panel under the image."""
        for widget in parent.winfo_children():
            widget.destroy()

        title = tk.Label(parent, text="Privacy Meter", font=("Arial", 18, "bold"), bg="white")
        title.pack(anchor="w")

        status_label = tk.Label(parent, text="Calculating...", font=("Arial", 14), bg="white", fg="#555555")
        status_label.pack(anchor="w", pady=(5, 0))

        canvas = tk.Canvas(parent, width=420, height=35, bg="white", highlightthickness=0)
        canvas.pack(pady=10)

        info_label = tk.Label(parent, text="Analyzing the latest edits...", font=("Arial", 12), bg="white", fg="#777777", wraplength=420, justify="left")
        info_label.pack(anchor="w")

        parent.status_label = status_label  # type: ignore[attr-defined]
        parent.canvas = canvas  # type: ignore[attr-defined]
        parent.info_label = info_label  # type: ignore[attr-defined]
        parent.current_image_path = image_path  # type: ignore[attr-defined]

        def worker():
            analysis = self.get_privacy_analysis(image_path)
            self.after(0, lambda: self.populate_privacy_meter(parent, analysis, image_path))

        threading.Thread(target=worker, daemon=True).start()

    def populate_privacy_meter(self, parent: tk.Frame, analysis: dict, image_path: str):
        """Update the meter UI with calculated analysis."""
        if getattr(parent, "current_image_path", None) != image_path:
            return
        if self.current_displayed_image_path != image_path:
            return

        status_label: tk.Label = getattr(parent, "status_label", None)
        info_label: tk.Label = getattr(parent, "info_label", None)
        canvas: tk.Canvas = getattr(parent, "canvas", None)

        if analysis.get("error"):
            if status_label:
                status_label.config(text="Unable to compute privacy score", fg="#c62828")
            if info_label:
                info_label.config(text=str(analysis["error"]), fg="#c62828")
            if canvas:
                canvas.delete("all")
            return

        try:
            score = int(analysis.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        risk = analysis.get("risk_level", "unknown").title()
        safe_flag = analysis.get("safe_to_share")
        descriptor = "Safe to share now" if safe_flag else "Needs more cleaning"

        if status_label:
            status_label.config(text=f"{score}/100 • {risk} risk", fg="#1b5e20" if safe_flag else "#e65100")

        if canvas:
            self.draw_privacy_meter_bar(canvas, score)

        reasons = analysis.get("reasons", [])
        if info_label:
            if reasons:
                info_label.config(
                    text=f"{descriptor}. Example issue: {reasons[0]}",
                    fg="#1b5e20" if safe_flag else "#e65100"
                )
            else:
                info_label.config(text=descriptor, fg="#1b5e20" if safe_flag else "#e65100")

    def draw_privacy_meter_bar(self, canvas: tk.Canvas, score: int):
        """Draw gradient bar with current score overlay."""
        canvas.delete("all")
        bar_height = 20
        bar_width = 400
        x0 = 10
        y0 = 5
        for i in range(bar_width):
            ratio = i / bar_width
            red = int(255 * (1 - ratio))
            green = int(255 * ratio)
            color = f"#{red:02x}{green:02x}20"
            canvas.create_line(x0 + i, y0, x0 + i, y0 + bar_height, fill=color)

        fill_width = max(0, min(bar_width, int(bar_width * (score / 100))))
        if fill_width < bar_width:
            canvas.create_rectangle(x0 + fill_width, y0, x0 + bar_width, y0 + bar_height, fill="#f5f5f5", outline="")

        canvas.create_rectangle(x0, y0, x0 + bar_width, y0 + bar_height, outline="#424242", width=2)
        canvas.create_line(x0 + fill_width, y0, x0 + fill_width, y0 + bar_height, fill="#212121", width=2)

    def get_privacy_analysis(self, image_path: str, force_refresh: bool = False) -> dict:
        """Return cached privacy analysis or compute a new one."""
        if not force_refresh and image_path in self.privacy_meter_cache:
            return self.privacy_meter_cache[image_path]
        try:
            with open(image_path, "rb") as file:
                data = file.read()
            analysis = analyze_image_bytes(data, Path(image_path).name)
            self.privacy_meter_cache[image_path] = analysis
            return analysis
        except Exception as exc:
            return {"error": str(exc)}

    def invalidate_privacy_cache(self, image_path: Optional[str] = None):
        """Remove cached privacy analysis entries."""
        if image_path:
            self.privacy_meter_cache.pop(image_path, None)
        else:
            self.privacy_meter_cache.clear()

    def rename_image(self):
        """Option 5: Rename image"""
        current_image_path = self.selected_files[self.current_image_index]
        current_name = Path(current_image_path).stem
        
        new_name = simpledialog.askstring("Rename Image", f"Enter new name (without extension):", initialvalue=current_name)
        
        if new_name:
            try:
                old_path = Path(current_image_path)
                new_path = old_path.parent / f"{new_name}{old_path.suffix}"
                
                # Handle duplicates
                counter = 1
                while new_path.exists():
                    new_path = old_path.parent / f"{new_name}_{counter}{old_path.suffix}"
                    counter += 1
                
                os.rename(current_image_path, str(new_path))
                self.invalidate_privacy_cache(current_image_path)
                self.selected_files[self.current_image_index] = str(new_path)
                
                messagebox.showinfo("Success", f"Image renamed to {new_path.name}")
                self.show_metadata_cleaning_page()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to rename:\n{e}")

    def image_done(self):
        """Option 6: Done with current image, move to next"""
        # Save current processed image
        current_image_path = self.selected_files[self.current_image_index]
        self.processed_images.append(current_image_path)
        
        self.current_image_index += 1
        self.show_metadata_cleaning_page()

    def show_download_page(self):
        """Final download page after all images processed"""
        self.current_displayed_image_path = None
        for widget in self.winfo_children():
            widget.destroy()

        # Background
        canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, bg="#f5f5f5", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Title
        title_label = tk.Label(
            self,
            text="Processing Complete!",
            font=("Arial", 32, "bold"),
            bg="#f5f5f5"
        )
        canvas.create_window(self.screen_width // 2, 150, anchor="center", window=title_label)

        subtitle_label = tk.Label(
            self,
            text=f"All {len(self.processed_images)} image(s) have been processed.",
            font=("Arial", 18),
            bg="#f5f5f5"
        )
        canvas.create_window(self.screen_width // 2, 220, anchor="center", window=subtitle_label)

        # Download buttons
        download_individual_btn = tk.Button(
            self,
            text="Download Individual Images",
            font=("Arial", 18, "bold"),
            bg="#2196F3",
            fg="white",
            padx=40,
            pady=20,
            command=self.download_individual
        )
        canvas.create_window(self.screen_width // 2, 350, anchor="center", window=download_individual_btn)

        download_compressed_btn = tk.Button(
            self,
            text="Download Compressed ZIP",
            font=("Arial", 18, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=40,
            pady=20,
            command=self.download_compressed
        )
        canvas.create_window(self.screen_width // 2, 450, anchor="center", window=download_compressed_btn)

        # Back to home button
        home_btn = tk.Button(
            self,
            text="Back to Home",
            font=("Arial", 14),
            bg="#757575",
            fg="white",
            padx=30,
            pady=10,
            command=self.reset_to_home
        )
        canvas.create_window(self.screen_width // 2, 550, anchor="center", window=home_btn)

    def reset_to_home(self):
        """Reset selections and return to home screen."""
        self.selected_files = []
        self.processed_images = []
        self.current_image_index = 0
        self.current_displayed_image_path = None
        self.invalidate_privacy_cache()
        self.cleanup_temp_workspace()
        self.show_home_page()

    def cleanup_temp_workspace(self):
        """Clear temporary working directory."""
        try:
            for item in self.temp_work_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        except Exception:
            pass

    def download_individual(self):
        """Open folder with individual processed images"""
        if not self.processed_images:
            messagebox.showinfo("No Images", "No processed images available.")
            return
        
        try:
            # Copy all processed images to a final output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_dir = OUTPUT_ROOT / "final" / timestamp
            final_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in self.processed_images:
                shutil.copy2(img_path, final_dir / Path(img_path).name)
            
            os.startfile(final_dir)  # type: ignore
            messagebox.showinfo("Success", f"Images saved to:\n{final_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare download:\n{e}")

    def download_compressed(self):
        """Create and download compressed ZIP"""
        if not self.processed_images:
            messagebox.showinfo("No Images", "No processed images available.")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_dir = OUTPUT_ROOT / "package" / timestamp
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images to package directory
            for img_path in self.processed_images:
                shutil.copy2(img_path, package_dir / Path(img_path).name)
            
            # Compress and create ZIP
            result = compress_and_package(
                [str(package_dir / Path(p).name) for p in self.processed_images],
                package_dir
            )
            
            zip_path = Path(result["zip_path"]) if result.get("zip_path") else None
            
            if zip_path and zip_path.exists():
                os.startfile(zip_path.parent)  # type: ignore
                messagebox.showinfo("Success", f"ZIP file created:\n{zip_path.name}\n\nLocation: {zip_path.parent}")
            else:
                messagebox.showerror("Error", "Failed to create ZIP file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create compressed package:\n{e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
