import os
import sys
import threading
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
from meta.modules.metadata_tools import read_exif, compute_privacy_score, remove_exif
from blurring.modules.blur_tools import detect_faces, detect_plates, _blur_regions, _blur_text_regions, _tesseract_available

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

        self.show_home_page()

    def show_home_page(self):
        """Initial home page with Get Started button"""
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

        # Left side - Image preview
        image_frame = tk.Frame(content_frame, bg="white", relief="sunken", bd=2)
        image_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

        try:
            img = Image.open(current_image_path)
            # Resize to fit screen while maintaining aspect ratio
            max_width = self.screen_width // 2 - 100
            max_height = self.screen_height - 200
            
            img.thumbnail((max_width, max_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            image_label = tk.Label(image_frame, image=photo, bg="white")
            image_label.image = photo  # Keep reference
            image_label.pack(padx=10, pady=10)
        except Exception as e:
            error_label = tk.Label(image_frame, text=f"Error loading image:\n{e}", fg="red", font=("Arial", 12))
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
            ("3. Blur Faces and Text", self.blur_faces_interactive),
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
            exif = read_exif(current_image_path)
            score, details = compute_privacy_score(exif)
            
            score_window = tk.Toplevel(self)
            score_window.title("Privacy Score")
            score_window.geometry("500x300")
            
            text_area = scrolledtext.ScrolledText(score_window, wrap=tk.WORD, width=60, height=15)
            text_area.pack(fill="both", expand=True, padx=10, pady=10)
            
            text_area.insert(tk.END, f"File: {Path(current_image_path).name}\n")
            text_area.insert(tk.END, "="*50 + "\n\n")
            text_area.insert(tk.END, f"Privacy Score: {score}/100\n")
            text_area.insert(tk.END, f"(Higher = more private)\n\n")
            
            if details.get("found_fields"):
                text_area.insert(tk.END, f"Sensitive fields found:\n")
                for field in details['found_fields']:
                    text_area.insert(tk.END, f"  - {field}\n")
            else:
                text_area.insert(tk.END, "No sensitive metadata detected.\n")
            
            text_area.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute privacy score:\n{e}")

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
            command=self.show_home_page
        )
        canvas.create_window(self.screen_width // 2, 550, anchor="center", window=home_btn)

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
                import shutil
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
            import shutil
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
