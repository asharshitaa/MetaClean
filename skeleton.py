import tkinter as tk
from PIL import Image, ImageTk

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Custom Tkinter Fullscreen App")
        self.attributes('-fullscreen', True)
        self.bind("<Escape>", lambda e: self.destroy())

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Load and resize background image to fit screen
        bg_image = Image.open("bg.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Main frame
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Canvas with bg
        self.canvas = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        # Get Started button
        getstarted_img = Image.open("getstarted.png")
        scale_factor = 0.7
        getstarted_img = getstarted_img.resize(
            (int(getstarted_img.width * scale_factor), int(getstarted_img.height * scale_factor)),
            Image.LANCZOS
        )
        self.getstarted_photo = ImageTk.PhotoImage(getstarted_img)

        self.getstarted_button = tk.Button(
            self.main_frame,
            image=self.getstarted_photo,
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg="#000000",
            activebackground="#000000",
            command=self.show_next_page
        )

        # Place Get Started button
        self.canvas.create_window(
            self.screen_width // 2,
            (self.screen_height // 2) + 300,
            anchor="center",
            window=self.getstarted_button
        )

        # Close button (X)
        close_button = tk.Button(
            self.main_frame,
            text="x",
            font=("Arial", 20, "bold"),
            fg="black",
            bg="white",
            activebackground="darkred",
            relief="flat",
            bd=0,
            command=self.destroy
        )
        self.canvas.create_window(self.screen_width - 60, 90, anchor="ne", window=close_button)

    def show_next_page(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        bg2_image = Image.open("bg2.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.bg2_photo = ImageTk.PhotoImage(bg2_image)

        canvas2 = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas2.pack(fill="both", expand=True)
        canvas2.create_image(0, 0, image=self.bg2_photo, anchor="nw")

        # Cancel & Next
        scale_small = 0.4
        cancel_img = Image.open("cancel.png").resize(
            (int(Image.open("cancel.png").width * scale_small),
             int(Image.open("cancel.png").height * scale_small)),
            Image.LANCZOS
        )
        next_img = Image.open("next.png").resize(
            (int(Image.open("next.png").width * scale_small),
             int(Image.open("next.png").height * scale_small)),
            Image.LANCZOS
        )
        self.cancel_photo = ImageTk.PhotoImage(cancel_img)
        self.next_photo = ImageTk.PhotoImage(next_img)

        cancel_btn = tk.Button(self.main_frame, image=self.cancel_photo, bd=0, relief="flat", bg="#ffffff", command=self.destroy)
        next_btn = tk.Button(self.main_frame, image=self.next_photo, bd=0, relief="flat", bg="#ffffff", command=self.next_btn)

        button_y = self.screen_height - 200
        canvas2.create_window(self.screen_width - 280, button_y, anchor="nw", window=cancel_btn)
        canvas2.create_window(self.screen_width - 160, button_y, anchor="nw", window=next_btn)

        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              activebackground="darkred", relief="flat", bd=0, command=self.destroy)
        canvas2.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)

    def next_btn(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        bg3_image = Image.open("bg3.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.bg3_photo = ImageTk.PhotoImage(bg3_image)

        canvas3 = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas3.pack(fill="both", expand=True)
        canvas3.create_image(0, 0, image=self.bg3_photo, anchor="nw")

        cancel_img = Image.open("cancel.png")
        next_img = Image.open("next.png")
        scale_small = 0.4
        cancel_img = cancel_img.resize(
            (int(cancel_img.width * scale_small), int(cancel_img.height * scale_small)),
            Image.LANCZOS
        )
        next_img = next_img.resize(
            (int(next_img.width * scale_small), int(next_img.height * scale_small)),
            Image.LANCZOS
        )
        self.cancel_photo = ImageTk.PhotoImage(cancel_img)
        self.next_photo = ImageTk.PhotoImage(next_img)

        cancel_btn = tk.Button(self.main_frame, image=self.cancel_photo, bd=0, relief="flat", bg="#ffffff",
                               command=self.destroy)
        next_btn = tk.Button(self.main_frame, image=self.next_photo, bd=0, relief="flat", bg="#ffffff",
                             command=self.show_main_page)

        button_y = self.screen_height - 120
        canvas3.create_window(self.screen_width - 280, button_y, anchor="nw", window=cancel_btn)
        canvas3.create_window(self.screen_width - 160, button_y, anchor="nw", window=next_btn)

        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              activebackground="darkred", relief="flat", bd=0, command=self.destroy)
        canvas3.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)

    def show_main_page(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        main_img = Image.open("main1.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.main_photo = ImageTk.PhotoImage(main_img)

        canvas_main = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas_main.pack(fill="both", expand=True)
        canvas_main.create_image(0, 0, image=self.main_photo, anchor="nw")

        # --- Top Buttons ---
        back_img = Image.open("back.png").resize((80, 40), Image.LANCZOS)
        help_img = Image.open("help.png").resize((80, 40), Image.LANCZOS)
        self.back_photo = ImageTk.PhotoImage(back_img)
        self.help_photo = ImageTk.PhotoImage(help_img)

        back_btn = tk.Button(self.main_frame, image=self.back_photo, bd=0, relief="flat", bg="#ffffff", command=self.__init__)
        help_btn = tk.Button(self.main_frame, image=self.help_photo, bd=0, relief="flat", bg="#ffffff")
        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              relief="flat", bd=0, command=self.destroy)

        canvas_main.create_window(80, 90, anchor="nw", window=back_btn)
        canvas_main.create_window(self.screen_width - 160, 90, anchor="ne", window=help_btn)
        canvas_main.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)

        # --- Clean Button ---
        clean_img = Image.open("clean.png").resize(
            (int(Image.open("clean.png").width * 0.4), int(Image.open("clean.png").height * 0.4)), Image.LANCZOS
        )
        self.clean_photo = ImageTk.PhotoImage(clean_img)
        canvas_main.create_window(
            self.screen_width // 2,
            (self.screen_height // 2) + 350,
            anchor="center",
            window=tk.Button(self.main_frame, image=self.clean_photo, bd=0, relief="flat", bg="#ffffff", command=self.clean)
        )

        # --- Four Middle Buttons ---
        button_imgs = ["addmore.png", "remove_clicked.png", "find.png", "score.png"]
        scale_mid = 0.3
        photos = []
        for path in button_imgs:
            img = Image.open(path)
            img = img.resize((int(img.width * scale_mid), int(img.height * scale_mid)), Image.LANCZOS)
            photos.append(ImageTk.PhotoImage(img))

        self.add_photo, self.remove_photo, self.find_photo, self.score_photo = photos
        button_y = (self.screen_height // 2) - 230
        spacing = 200
        start_x = (self.screen_width // 2) - (spacing * 1.5)
        for i, img in enumerate([self.add_photo, self.remove_photo, self.find_photo, self.score_photo]):
            canvas_main.create_window(start_x + (i * spacing), button_y, anchor="center",
                                      window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff"))
            
        # --- Three Smaller Buttons (Choose, Select All, Delete) ---
        small_imgs = ["choose.png", "selectall.png", "delete.png"]
        scale_small = 0.2
        small_photos = []
        for path in small_imgs:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small)), Image.LANCZOS )
            small_photos.append(ImageTk.PhotoImage(img))
        self.choose_photo, self.selectall_photo, self.delete_photo = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) - 490 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo, self.selectall_photo, self.delete_photo]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )
        
        small_imgs3 = ["choose.png", "selectall.png", "delete.png"]
            
        scale_small3 = 0.2
        small_photos3 = []
        for path in small_imgs3:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small3)), Image.LANCZOS )
            small_photos3.append(ImageTk.PhotoImage(img))
        self.choose_photo3, self.selectall_photo3, self.delete_photo3 = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) + 380 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo3, self.selectall_photo3, self.delete_photo3]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )
        
        small_imgs2 = ["choose.png", "selectall.png", "delete.png"]
        scale_small2 = 0.2
        small_photos2 = []
        for path in small_imgs2:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small2)), Image.LANCZOS )
            small_photos2.append(ImageTk.PhotoImage(img))
        self.choose_photo2, self.selectall_photo2, self.delete_photo2 = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) -50 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo2, self.selectall_photo2, self.delete_photo2]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )

    def clean(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        main_img = Image.open("main1.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.main_photo = ImageTk.PhotoImage(main_img)

        canvas_main = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas_main.pack(fill="both", expand=True)
        canvas_main.create_image(0, 0, image=self.main_photo, anchor="nw")

        # --- Top Buttons ---
        back_img = Image.open("back.png").resize((80, 40), Image.LANCZOS)
        help_img = Image.open("help.png").resize((80, 40), Image.LANCZOS)
        self.back_photo = ImageTk.PhotoImage(back_img)
        self.help_photo = ImageTk.PhotoImage(help_img)

        back_btn = tk.Button(self.main_frame, image=self.back_photo, bd=0, relief="flat", bg="#ffffff", command=self.__init__)
        help_btn = tk.Button(self.main_frame, image=self.help_photo, bd=0, relief="flat", bg="#ffffff")
        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              relief="flat", bd=0, command=self.destroy)

        canvas_main.create_window(80, 90, anchor="nw", window=back_btn)
        canvas_main.create_window(self.screen_width - 160, 90, anchor="ne", window=help_btn)
        canvas_main.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)

        # --- Two Buttons (Download, Compress) ---
        clean_img = Image.open("clean.png")
        scale_clean = 0.4
        target_height = int(clean_img.height * scale_clean)

        def resize_by_height(img_path, target_h):
            img = Image.open(img_path)
            aspect_ratio = img.width / img.height
            new_width = int(target_h * aspect_ratio)
            return img.resize((new_width, target_h), Image.LANCZOS)

        download_img = resize_by_height("download.png", target_height)
        compress_img = resize_by_height("compress.png", target_height)

        self.download_photo = ImageTk.PhotoImage(download_img)
        self.compress_photo = ImageTk.PhotoImage(compress_img)

        button_y = (self.screen_height // 2) + 350
        spacing = 250
        start_x = (self.screen_width // 2) - (spacing // 2)

        canvas_main.create_window(start_x, button_y, anchor="center",
                                  window=tk.Button(self.main_frame, image=self.download_photo, bd=0, relief="flat", bg="#ffffff"))
        canvas_main.create_window(start_x + spacing, button_y, anchor="center",
                                  window=tk.Button(self.main_frame, image=self.compress_photo, bd=0, relief="flat", bg="#ffffff"))
        
        # --- Four Middle Buttons (Add, Remove, Find, Score) ---
        button_imgs = ["addmore.png", "remove_clicked.png", "find.png", "score.png"]
        scale_mid = 0.3
        photos = []
        for path in button_imgs:
            img = Image.open(path)
            img = img.resize(
                (int(img.width * scale_mid), int(img.height * scale_mid)),
                Image.LANCZOS
            )
            photos.append(ImageTk.PhotoImage(img))

        self.add_photo, self.remove_photo, self.find_photo, self.score_photo = photos
        button_y = (self.screen_height // 2) - 230
        spacing = 200
        start_x = (self.screen_width // 2) - (spacing * 1.5)
        for i, img in enumerate([self.add_photo, self.remove_photo, self.find_photo, self.score_photo]):
            canvas_main.create_window(
                start_x + (i * spacing),
                button_y,
                anchor="center",
                window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff")
            )

        # --- Three Delete Buttons ---
        delete_img = Image.open("delete.png")
        scale_small = 0.2
        delete_img = delete_img.resize((int(delete_img.width * scale_small), int(delete_img.height * scale_small)), Image.LANCZOS)
        self.delete_photo = ImageTk.PhotoImage(delete_img)

        delete_y = (self.screen_height // 2) + 60
        start_x = (self.screen_width // 2) - 290
        spacing = 440

        for i in range(3):
            canvas_main.create_window(
                start_x + (i * spacing),
                delete_y,
                anchor="center",
                window=tk.Button(self.main_frame, image=self.delete_photo, bd=0, relief="flat", bg="#ffffff")
            )
    
    def find(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        main_img = Image.open("main1.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.main_photo = ImageTk.PhotoImage(main_img)

        canvas_main = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas_main.pack(fill="both", expand=True)
        canvas_main.create_image(0, 0, image=self.main_photo, anchor="nw")

        # --- Top Buttons ---
        back_img = Image.open("back.png").resize((80, 40), Image.LANCZOS)
        help_img = Image.open("help.png").resize((80, 40), Image.LANCZOS)
        self.back_photo = ImageTk.PhotoImage(back_img)
        self.help_photo = ImageTk.PhotoImage(help_img)

        back_btn = tk.Button(self.main_frame, image=self.back_photo, bd=0, relief="flat", bg="#ffffff", command=self.__init__)
        help_btn = tk.Button(self.main_frame, image=self.help_photo, bd=0, relief="flat", bg="#ffffff")
        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              relief="flat", bd=0, command=self.destroy)

        canvas_main.create_window(80, 90, anchor="nw", window=back_btn)
        canvas_main.create_window(self.screen_width - 160, 90, anchor="ne", window=help_btn)
        canvas_main.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)

        # --- Clean Button ---
        clean_img = Image.open("clean.png").resize(
            (int(Image.open("clean.png").width * 0.4), int(Image.open("clean.png").height * 0.4)), Image.LANCZOS
        )
        self.clean_photo = ImageTk.PhotoImage(clean_img)
        canvas_main.create_window(
            self.screen_width // 2,
            (self.screen_height // 2) + 350,
            anchor="center",
            window=tk.Button(self.main_frame, image=self.clean_photo, bd=0, relief="flat", bg="#ffffff", command=self.clean)
        )

        # --- Four Middle Buttons ---
        button_imgs = ["addmore.png", "remove.png", "find_clicked.png", "score.png"]
        scale_mid = 0.3
        photos = []
        for path in button_imgs:
            img = Image.open(path)
            img = img.resize((int(img.width * scale_mid), int(img.height * scale_mid)), Image.LANCZOS)
            photos.append(ImageTk.PhotoImage(img))

        self.add_photo, self.remove_photo, self.find_photo, self.score_photo = photos
        button_y = (self.screen_height // 2) - 230
        spacing = 200
        start_x = (self.screen_width // 2) - (spacing * 1.5)
        for i, img in enumerate([self.add_photo, self.remove_photo, self.find_photo, self.score_photo]):
            canvas_main.create_window(start_x + (i * spacing), button_y, anchor="center",
                                      window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff"))
            
        # --- Three Smaller Buttons (Choose, Select All, Delete) ---
        small_imgs = ["choose.png", "selectall.png", "delete.png"]
        scale_small = 0.2
        small_photos = []
        for path in small_imgs:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small)), Image.LANCZOS )
            small_photos.append(ImageTk.PhotoImage(img))
        self.choose_photo, self.selectall_photo, self.delete_photo = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) - 490 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo, self.selectall_photo, self.delete_photo]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )
        
        small_imgs3 = ["choose.png", "selectall.png", "delete.png"]
            
        scale_small3 = 0.2
        small_photos3 = []
        for path in small_imgs3:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small3)), Image.LANCZOS )
            small_photos3.append(ImageTk.PhotoImage(img))
        self.choose_photo3, self.selectall_photo3, self.delete_photo3 = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) + 380 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo3, self.selectall_photo3, self.delete_photo3]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )
        
        small_imgs2 = ["choose.png", "selectall.png", "delete.png"]
        scale_small2 = 0.2
        small_photos2 = []
        for path in small_imgs2:
            img = Image.open(path)
            img = img.resize( (int(img.width * scale_small), int(img.height * scale_small2)), Image.LANCZOS )
            small_photos2.append(ImageTk.PhotoImage(img))
        self.choose_photo2, self.selectall_photo2, self.delete_photo2 = small_photos
        small_button_y = (self.screen_height // 2) + 60 # below center
        small_start_x = (self.screen_width//2) -50 # right side
        small_spacing = 100 # close together
        buttons = [self.choose_photo2, self.selectall_photo2, self.delete_photo2]
        for i, img in enumerate(buttons):
            canvas_main.create_window( small_start_x + (i * small_spacing), small_button_y, anchor="center",
                                       window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff") )
            
    
    def score(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        main_img = Image.open("main1.png").resize((self.screen_width, self.screen_height), Image.LANCZOS)
        self.main_photo = ImageTk.PhotoImage(main_img)

        canvas_main = tk.Canvas(self.main_frame, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        canvas_main.pack(fill="both", expand=True)
        canvas_main.create_image(0, 0, image=self.main_photo, anchor="nw")

        # --- Top Buttons ---
        back_img = Image.open("back.png").resize((80, 40), Image.LANCZOS)
        help_img = Image.open("help.png").resize((80, 40), Image.LANCZOS)
        self.back_photo = ImageTk.PhotoImage(back_img)
        self.help_photo = ImageTk.PhotoImage(help_img)

        back_btn = tk.Button(self.main_frame, image=self.back_photo, bd=0, relief="flat", bg="#ffffff", command=self.__init__)
        help_btn = tk.Button(self.main_frame, image=self.help_photo, bd=0, relief="flat", bg="#ffffff")
        close_btn = tk.Button(self.main_frame, text="x", font=("Arial", 20, "bold"), fg="black", bg="white",
                              relief="flat", bd=0, command=self.destroy)

        canvas_main.create_window(80, 90, anchor="nw", window=back_btn)
        canvas_main.create_window(self.screen_width - 160, 90, anchor="ne", window=help_btn)
        canvas_main.create_window(self.screen_width - 60, 90, anchor="ne", window=close_btn)
        
        
        
        # --- Four Middle Buttons (Add, Remove, Find, Score) ---
        button_imgs = ["addmore.png", "remove.png", "find.png", "score_clicked.png"]
        scale_mid = 0.3
        photos = []
        for path in button_imgs:
            img = Image.open(path)
            img = img.resize(
                (int(img.width * scale_mid), int(img.height * scale_mid)),
                Image.LANCZOS
            )
            photos.append(ImageTk.PhotoImage(img))

        self.add_photo, self.remove_photo, self.find_photo, self.score_photo = photos
        button_y = (self.screen_height // 2) - 230
        spacing = 200
        start_x = (self.screen_width // 2) - (spacing * 1.5)
        for i, img in enumerate([self.add_photo, self.remove_photo, self.find_photo, self.score_photo]):
            canvas_main.create_window(
                start_x + (i * spacing),
                button_y,
                anchor="center",
                window=tk.Button(self.main_frame, image=img, bd=0, relief="flat", bg="#ffffff")
            )

        # --- Three Delete Buttons ---
        delete_img = Image.open("delete.png")
        scale_small = 0.2
        delete_img = delete_img.resize((int(delete_img.width * scale_small), int(delete_img.height * scale_small)), Image.LANCZOS)
        self.delete_photo = ImageTk.PhotoImage(delete_img)

        delete_y = (self.screen_height // 2) + 60
        start_x = (self.screen_width // 2) - 290
        spacing = 440

        for i in range(3):
            canvas_main.create_window(
                start_x + (i * spacing),
                delete_y,
                anchor="center",
                window=tk.Button(self.main_frame, image=self.delete_photo, bd=0, relief="flat", bg="#ffffff")
            )


if __name__ == "__main__":
    app = App()
    app.mainloop()
