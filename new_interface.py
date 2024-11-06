import tkinter as tk
from tkinter import Button, OptionMenu, StringVar
import threading
import cv2
from functions import *
import numpy as np
from tkinter import ttk  # Required for Combobox
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from functools import partial
from functions import * 
import subprocess


def open_first_window():
    color_dict = {
    1: (np.array([0, 120, 100]), np.array([40, 255, 255]), (255, 0, 0)),  # Blue
    2: (np.array([40, 50, 80]), np.array([80, 100, 255]), (0, 255, 0)),   # Green
    3: (np.array([100, 30, 140]), np.array([180, 80, 200]), (0, 255, 255))  # Yellow
    }


    color_ranges = {
    1: (np.array([0, 120, 100]), np.array([40, 255, 255])),  # Blue
        2: (np.array([40, 50, 80]), np.array([80, 100, 255])),   # Green
        3: (np.array([100, 30, 140]), np.array([180, 80, 200]))  # Yellow
        # Add more color ranges as needed
    }


    # Function to handle filter button clicks
    def filter_button_click(filter_type):
        global current_filter
        current_filter = filter_type



    # Function to handle the start button click
    def start_button_click():
        global video_started
        video_started = True  # Set the flag to indicate video is started
        root.after(10, video_loop)  # Start the video loop
        


    # Function to handle stop button click
    def stop_button_click():
        global video_started, cap

        video_started = False

        # Release the video capture
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        # Re-initialize the video capture for a new stream
        cap = cv2.VideoCapture(0)


    # Set the initial filter type
    # current_filter = 1
    
    first_window = tk.Toplevel(root)
    first_window.title("Object Color Detection")
    first_window.geometry("400x400")

    # Set the initial filter type
    # current_filter = 1
    video_started = False

    # Create filter buttons
    button_color_detect = Button(first_window, text="color detection simple", command=lambda: filter_button_click(0))
    button_median = Button(first_window, text="amlioration using Median Filter", command=lambda: filter_button_click(1))
    button_gaussian = Button(first_window, text="amlioration using Gaussian Filter", command=lambda: filter_button_click(2))
    button_sub_background =  Button(first_window, text="substracting background amelioration", command=lambda: filter_button_click(4))

    # Dropdown list for color selection
    color_var = StringVar(first_window)
    color_var.set("Select Color")  # Default text
    color_dropdown = OptionMenu(first_window, color_var, "Blue", "Green", "Yellow")
    color_dropdown.pack(pady=10)

    button_color = Button(first_window, text="color amelioration", command=lambda: filter_button_click(3))

    # Pack filter buttons into the window
    button_color_detect.pack(pady=10)
    button_median.pack(pady=10)
    button_gaussian.pack(pady=10)
    button_sub_background.pack(pady=10)
    button_color.pack(pady=10)

    # Create start and stop buttons
    button_start = Button(first_window, text="Start Video", command=start_button_click)
    button_stop = Button(first_window, text="Stop Video", command=stop_button_click)
    button_start.pack(pady=5)
    button_stop.pack(pady=5)

    # OpenCV setup
    cap = cv2.VideoCapture(0)
    vois = 3  # Set the neighborhood size for the median filter
    

    def video_loop():
        global video_started

        if video_started:
            ret, frame = cap.read()
            ret, background = cap.read()
            color_id_mapping = {"Blue": 1, "Green": 2, "Yellow": 3}
            selected_color = color_var.get()
            color_id = color_id_mapping.get(selected_color, 1)

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            if current_filter == 0:
                result = detect_color(frame, color_id, color_ranges)
            else:
                # Apply the selected filter
                if current_filter == 1:
                    frame_smoothed = filtreMed(frame, vois)
                    result = detect_color(frame_smoothed, color_id, color_ranges)
                elif current_filter == 2:
                    frame_smoothed = gaussianFilter2(frame, sigma=2)
                    result = detect_color(frame_smoothed, color_id, color_ranges)
                elif current_filter == 3:
                    frame_smoothed, result = detect_and_draw_color(frame, color_dict, color_id)
                elif current_filter == 4:
                     subtracted_result = subtract_background(frame, background)
                     result = detect_color_sub(frame, color_id, color_ranges,subtracted_result)

                
               

            cv2.imshow('Original', frame)
            cv2.imshow('Result', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_started = False

            # Schedule the next iteration
            root.after(10, video_loop)
        else:
            # Release resources when the video is stopped
            cap.release()
            cv2.destroyAllWindows()

    # Start the Tkinter main loop
    first_window.mainloop()
    
    label = tk.Label(first_window)
    label.pack()

def open_second_window():
    second_window = tk.Toplevel(root)
    second_window.title("Filtres en fonctions de base")
    second_window.geometry("400x300")
    label = tk.Label(second_window)
    label.grid(row=0, column=0, padx=10, pady=10)
    class ImageProcessorApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Image Processor")

            # Variables
            image_frame = tk.Frame(root)
            image_frame.grid(row=2, column=1, padx=10, pady=10, sticky="ne")  # Place the image frame to the right

        # Add label to the image frame
            
            self.original_image = None
            self.filtered_image = None
            self.image_label = tk.Label(image_frame)
            self.image_label.pack()
            self.filter_entries = {}

            # Radio buttons for upload picture or use camera
            self.upload_var = tk.IntVar()
            self.upload_var.set(1)
            upload_radiobutton = tk.Radiobutton(root, text="Upload Picture (priviliege small images for better visualization)", variable=self.upload_var, value=1, command=self.upload_picture)
            upload_radiobutton.grid(row=0, column=0, columnspan=2)
            
            # Create a frame for filter buttons
            filter_frame = tk.Frame(root)
            filter_frame.grid(row=2, column=0, padx=10, pady=10, sticky="w")
            # Add label to the filter frame
            filter_label = tk.Label(filter_frame, text="Filter Options")
            filter_label.grid(row=0, column=0, columnspan=5, pady=5)

            # Button 1: Mean
            blur_button = tk.Button(filter_frame, text="Mean", command=partial(self.apply_filter, "Mean"))
            blur_button.grid(row=2, column=0, padx=5, pady=5, sticky="w")

            label = tk.Label(filter_frame, text="V:")
            label.grid(row=2, column=1, padx=5, pady=5, sticky="e")

            entry = tk.Entry(filter_frame)
            entry.grid(row=2, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Mean"] = [entry]
            
            # Button 2: Median
            blur_button = tk.Button(filter_frame, text="Median", command=partial(self.apply_filter, "Median"))
            blur_button.grid(row=3, column=0, padx=5, pady=5, sticky="w")
            # entry & label
            label2 = tk.Label(filter_frame, text="V:")
            label2.grid(row=3, column=1, padx=5, pady=5, sticky="e")
            entry2 = tk.Entry(filter_frame)
            entry2.grid(row=3, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Median"] = [entry2]

            # Button 3: Binarization
            contour_button = tk.Button(filter_frame, text="Binarization", command=partial(self.apply_filter, "Binarization"))
            contour_button.grid(row=4, column=0, padx=5, pady=5, sticky="w")
            # entry 1
            label3 = tk.Label(filter_frame, text="Threshold:")
            label3.grid(row=4, column=1, padx=5, pady=5, sticky="e")
            entry3 = tk.Entry(filter_frame)
            entry3.grid(row=4, column=2, padx=5, pady=5, sticky="w")
            # entry 2
            label4 = tk.Label(filter_frame, text="Algo type:")
            label4.grid(row=4, column=3, padx=5, pady=5, sticky="e")
            # Create a Combobox for the list of choices
            choices = ['BIN', 'BIN_INV', 'TRUNC', 'ZERO', 'ZERO_INV']  
            self.combo = ttk.Combobox(filter_frame, values=choices)
            self.combo.set(choices[0])  # Set the default choice
            self.combo.grid(row=4, column=4, padx=10, pady=5, sticky="w")
            self.filter_entries["Binarization"] = [entry3, self.combo]
            
            # Button 3: Gaussian
            edge_button = tk.Button(filter_frame, text="Gaussian", command=partial(self.apply_filter, "Gaussian"))
            edge_button.grid(row=5, column=0, padx=5, pady=5, sticky="w")
            gl_label = tk.Label(filter_frame, text="Kernel size:")
            gl_label.grid(row=5, column=1, padx=5, pady=5, sticky="e")
            gl_entry = tk.Entry(filter_frame)
            gl_entry.grid(row=5, column=2, padx=5, pady=5, sticky="w")
            
            gl_label2 = tk.Label(filter_frame, text="Sigma:")
            gl_label2.grid(row=5, column=3, padx=5, pady=5, sticky="e")
            gl_entry2 = tk.Entry(filter_frame)
            gl_entry2.grid(row=5, column=4, padx=5, pady=5, sticky="w")
            self.filter_entries["Gaussian"] = [gl_entry, gl_entry2]

            # Button 4: Laplacian
            lp_button = tk.Button(filter_frame, text="Laplacian", command=partial(self.apply_filter, "Laplacian"))
            lp_button.grid(row=6, column=0, padx=5, pady=5, sticky="w")
            self.filter_entries["Laplacian"] = []

            
            # Button 5: Erosion
            ero_button = tk.Button(filter_frame, text="Erosion", command=partial(self.apply_filter, "Erosion"))
            ero_button.grid(row=7, column=0, padx=5, pady=5, sticky="w")

            label5 = tk.Label(filter_frame, text="Kernel size:")
            label5.grid(row=7, column=1, padx=5, pady=5, sticky="e")
            entry5 = tk.Entry(filter_frame)
            entry5.grid(row=7, column=2, padx=5, pady=5, sticky="w")
            # Create a Combobox for the list of choices
            # entry 2
            label51 = tk.Label(filter_frame, text="Kernel type:")
            label51.grid(row=7, column=3, padx=5, pady=5, sticky="e")
            choices2 = ['RECT', 'CROSS', 'ELLIPSE']  # Add your choices here
            self.combo2 = ttk.Combobox(filter_frame, values=choices)
            self.combo2.set(choices2[0])  # Set the default choice
            self.combo2.grid(row=7, column=4, padx=10, pady=5, sticky="w")
            self.filter_entries["Erosion"] = [entry5, self.combo2]
            
            # Button 6: Dilation
            dil_button = tk.Button(filter_frame, text="Dilation", command=partial(self.apply_filter, "Dilation"))
            dil_button.grid(row=8, column=0, padx=5, pady=5, sticky="w")

            label6 = tk.Label(filter_frame, text="Kernel size:")
            label6.grid(row=8, column=1, padx=5, pady=5, sticky="e")
            
            entry6 = tk.Entry(filter_frame)
            entry6.grid(row=8, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Dilation"] = [entry6, self.combo2]
            
            # Button 7: Opening
            op_button = tk.Button(filter_frame, text="Opening", command=partial(self.apply_filter, "Opening"))
            op_button.grid(row=9, column=0, padx=5, pady=5, sticky="w")

            label7 = tk.Label(filter_frame, text="Kernel size:")
            label7.grid(row=9, column=1, padx=5, pady=5, sticky="e")

            entry7 = tk.Entry(filter_frame)
            entry7.grid(row=9, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Opening"] = [entry7, self.combo2]
            
            # Button 8: Closing
            cl_button = tk.Button(filter_frame, text="Closing", command=partial(self.apply_filter, "Closing"))
            cl_button.grid(row=10, column=0, padx=5, pady=5, sticky="w")

            label8 = tk.Label(filter_frame, text="Kernel size:")
            label8.grid(row=10, column=1, padx=5, pady=5, sticky="e")

            entry8 = tk.Entry(filter_frame)
            entry8.grid(row=10, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Closing"] = [entry8, self.combo2]
            
            # Button 9: Motion
            cl_button = tk.Button(filter_frame, text="Motion", command=partial(self.apply_filter, "Motion"))
            cl_button.grid(row=11, column=0, padx=5, pady=5, sticky="w")

            label9 = tk.Label(filter_frame, text="Kernel size:")
            label9.grid(row=11, column=1, padx=5, pady=5, sticky="e")
            entry9 = tk.Entry(filter_frame)
            entry9.grid(row=11, column=2, padx=5, pady=5, sticky="w")
            self.filter_entries["Motion"] = [entry9]
            
            # Button 10: Emboss
            em_button = tk.Button(filter_frame, text="Emboss", command=partial(self.apply_filter, "Emboss"))
            em_button.grid(row=12, column=0, padx=5, pady=5, sticky="w")
            self.filter_entries["Emboss"] = []

            
            # Button 11: Symmetric
            sym_button = tk.Button(filter_frame, text="Symmetric", command=partial(self.apply_filter, "Symmetric"))
            sym_button.grid(row=13, column=0, padx=5, pady=5, sticky="w")
            self.filter_entries["Symmetric"] = []



        def upload_picture(self):
            file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
            if file_path:
                self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                self.display_original_image()

        def apply_filter(self, filter_name):
            if self.original_image is not None:
                
                self.image = self.original_image.copy()
                if self.filter_entries[filter_name]:
                    entry_values = [entry.get() for entry in self.filter_entries[filter_name]]
                if filter_name == "Mean":
                    voisinnage = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    self.image = meanFilter(self.image, voisinnage)
                    
                elif filter_name == "Median":
                    voisinnage = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    self.image = medFilter(self.image, voisinnage)
                    
                elif filter_name == "Binarization":
                    choices = ['BIN', 'BIN_INV', 'TRUNC', 'TO_ZERO', 'TO_ZERO_INV']  
                    threshold = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    algo = choices.index(self.combo.get())
                    self.image = binFilter(self.image, threshold, algo)

                elif filter_name == "Gaussian":
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    sigma = int(entry_values[1]) if entry_values[1].isdigit() else 1
                    self.image = gaussianFilter(self.image, gaussianKernel(kernel_size, sigma))
                    
                elif filter_name == "Laplacian":
                    self.image = laplacianFilter(self.image)
                    
                elif filter_name == "Erosion":
                    choices2 = ['RECT', 'CROSS', 'ELLIPSE'] 
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    kernel_type = choices2.index(self.combo2.get())
                    self.image = erosionFilter(self.image, kernelInit(kernel_type, kernel_size))
                    
                elif filter_name == "Dilation":
                    choices2 = ['RECT', 'CROSS', 'ELLIPSE']  
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    kernel_type = choices2.index(self.combo2.get())
                    self.image = dilationFilter(self.image, kernelInit(kernel_type, kernel_size))
                    
                elif filter_name == "Opening":
                    choices2 = ['RECT', 'CROSS', 'ELLIPSE']  
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    kernel_type = choices2.index(self.combo2.get())
                    self.image = openingFilter(self.image, kernelInit(kernel_type, kernel_size))
                    
                elif filter_name == "Closing":
                    choices2 = ['RECT', 'CROSS', 'ELLIPSE']  
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    kernel_type = choices2.index(self.combo2.get())
                    self.image = closingFilter(self.image, kernelInit(kernel_type, kernel_size))
                    
                elif filter_name == "Motion":
                    kernel_size = int(entry_values[0]) if entry_values[0].isdigit() else 1
                    self.image = motionFilter(self.image, kernel_size)
                    
                elif filter_name == "Emboss":
                    self.image = embossFilter(self.image)
                    
                elif filter_name == "Symmetric":
                    self.image = symmetricFilter(self.image)          
                
                self.filtered_image = self.image.copy()
                self.display_image()
                
        # Function to handle filter button clicks
        def filter_button_click(filter_type):
            global current_filter
            current_filter = filter_type
            
        def display_original_image(self):
            if self.original_image is not None:
                img_tk = self.convert_image_to_tkinter(self.original_image)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.image_label.grid(row=0, column=0, padx=10, pady=10) 

        def display_image(self):
            if self.filtered_image is not None:
                img_tk = self.convert_image_to_tkinter(self.filtered_image)
                self.image_label.config(image=img_tk)
                self.image_label.image = img_tk
                self.image_label.grid(row=0, column=0, padx=10, pady=10)  

        def convert_image_to_tkinter(self, image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            img_tk = ImageTk.PhotoImage(image=image)
            return img_tk

    app = ImageProcessorApp(second_window)
    second_window.mainloop()
    
    
    
    
def open_third_window():
    invisibility_cloak()

def open_fourth_window():
    def select_background_image():
        global background_image_path
        background_image_path = filedialog.askopenfilename(title="Select a background image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])

    def start_green_screen():
        global background_image_path
        if background_image_path:
            green_screen(background_image_path)
        else:
            print("Please select a background image.")

    background_image_path = None

    fourth_window = tk.Toplevel(root)
    fourth_window.title("Green Screen")
    fourth_window.geometry("400x200")

    select_button = tk.Button(fourth_window, text="Select Background Image", command=select_background_image)
    select_button.pack(pady=10)

    start_button = tk.Button(fourth_window, text="Start Green Screen", command=start_green_screen)
    start_button.pack(pady=10)

def open_fifth_window():
    subprocess.run(["python", "Part2/game.py"])



root = tk.Tk()
root.title("Main Window")
root.geometry("800x600")
root.configure(bg='white')

label = tk.Label(root, text="Projet Vision", font=('Helvetica', 40))
label.pack(pady=50)
label.config(bg=root.cget('bg'))  # Set label background same as root background

# Create buttons to open each specific window
button1 = tk.Button(root, text="Filtres en fonctions de base", command=open_second_window , font=('Helvetica', 15), bg='#ADD8E6')
button1.pack(pady=10)

button2 = tk.Button(root, text="Object Color Detection", command=open_first_window , font=('Helvetica', 15),bg='#98FB98')
button2.pack(pady=10)

button3 = tk.Button(root, text="Invisibility Cloak", command=open_third_window , font=('Helvetica', 15))
button3.pack(pady=10)

button4 = tk.Button(root, text="Green Screen", command=open_fourth_window , font=('Helvetica', 15), bg='#FFDAB9')
button4.pack(pady=10)

button5 = tk.Button(root, text="Brick racing game" , command=open_fifth_window ,font=('Helvetica', 15), bg='#E6E6FA')
button5.pack(pady=10)

root.mainloop()
