import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import ImageTk, Image

class ImageCaptionEditor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Caption Editor")
        self.root.configure(bg="#252626")
        self.folder_path = filedialog.askdirectory()
        self.image_files = [f for f in os.listdir(self.folder_path) if f.endswith((".jpg", ".png"))]
        self.current_image_index = 0
        self.current_image = None
        self.current_caption = None

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side="left")
        self.image_label.configure(borderwidth=0, highlightthickness=0)
        self.image_label.pack(side="left", padx=(0, 10))
        # create the image listbox
        self.image_listbox = tk.Listbox(self.root)
        self.image_listbox.configure(bg="#252626", fg="white")
        self.image_listbox.pack(side="right", fill="y")
        self.image_listbox.configure(borderwidth=0, highlightthickness=0)

        # add image filenames to the listbox
        for image_file in self.image_files:
            self.image_listbox.insert("end", image_file)
        
        # bind listbox select event to select_image_from_listbox method
        self.image_listbox.bind("<<ListboxSelect>>", self.select_image_from_listbox)

        self.caption_text = tk.Text(self.root)
        self.caption_text.configure(bg="#252626", fg="white", insertbackground="white")
        self.caption_text.pack(side="right")
        self.caption_text.configure(insertbackground="white", wrap="word")
        self.caption_text.pack(side="right")
        self.caption_text.configure(borderwidth=0, highlightthickness=0)

        self.image_number_label = tk.Label(self.root)
        self.image_number_label.pack(side="bottom")
        self.image_number_label.bind("<Button-1>", self.select_image)

        self.root.bind("<Down>", self.previous_image)
        self.root.bind("<Up>", self.next_image)
        self.root.bind("<Control-s>", self.save_caption)

        self.load_image()
    
    def select_image_from_listbox(self, event):
        selection = event.widget.curselection()
        if selection:
            self.current_image_index = selection[0]
            self.load_image()


    def load_image(self):
        image_file = self.image_files[self.current_image_index]
        image_path = os.path.join(self.folder_path, image_file)
        self.current_image = ImageTk.PhotoImage(Image.open(image_path))
        self.image_label.config(image=self.current_image)

        caption_file = os.path.splitext(image_file)[0] + ".txt"
        caption_path = os.path.join(self.folder_path, caption_file)
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                caption_text = f.read()
                self.caption_text.delete("1.0", "end")
                self.caption_text.insert("end", caption_text)
                self.current_caption = caption_path
        else:
            self.caption_text.delete("1.0", "end")
            self.current_caption = None

        image_number_text = f"{self.current_image_index + 1}/{len(self.image_files)}"
        self.image_number_label.config(text=image_number_text)

    def previous_image(self, event):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def next_image(self, event):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    def save_caption(self, event):
        if not self.current_caption:
            image_file = self.image_files[self.current_image_index]
            caption_file = os.path.splitext(image_file)[0] + ".txt"
            caption_path = os.path.join(self.folder_path, caption_file)
            self.current_caption = caption_path

        with open(self.current_caption, "w") as f:
            caption_text = self.caption_text.get("1.0", "end")
            f.write(caption_text)

    def select_image(self, event):
        image_number = simpledialog.askinteger("Select Image", "Enter image number:", parent=self.root, minvalue=1, maxvalue=len(self.image_files))
        if image_number:
            self.current_image_index = image_number - 1
            self.load_image()

if __name__ == "__main__":
    editor = ImageCaptionEditor()
    editor.root.mainloop()