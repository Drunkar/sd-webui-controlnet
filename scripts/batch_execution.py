import copy
import os

import cv2
import gradio as gr
import modules.scripts as scripts

from modules import images, shared
from modules.processing import process_images


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def save_images(path, image_list, name):
    for i, image in enumerate(image_list):
        images.save_image(image, path, f"{name}_{i}")
    

class Script(scripts.Script):  
    
    def title(self):
        return "controlnet batch"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        # How the script's is displayed in the UI. See https://gradio.app/docs/#components
        # for the different UI components you can use and how to create them.
        # Most UI components can return a value, such as a boolean for a checkbox.
        # The returned values are passed to the run method as parameters.
        
        ctrls_group = ()

        with gr.Group():
            with gr.Accordion("ControlNet-Batch", open = False):
                batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="controlnet_batch_input_dir")
        ctrls_group += (batch_input_dir,)

        return ctrls_group

    def run(self, p, *args):
        # This is where the additional processing is implemented. The parameters include
        # self, the model object "p" (a StableDiffusionProcessing class, see
        # processing.py), and the parameters returned by the ui method.
        # Custom functions can be defined here, and additional libraries can be imported 
        # to be used in processing. The return value should be a Processed object, which is
        # what is returned by the process_images method.
        batch_input_dir = args[0]

        input_images = load_images_from_folder(batch_input_dir)
        output_image_list = []

        for image in input_images:
            copy_p = copy.copy(p)
            copy_p.control_net_input_image = [image]
            proc = process_images(copy_p)
            img = proc.images[0]
            output_image_list.append(img)
            copy_p.close()
        
        return proc