import os
import time
import sys
import numpy as np
from glob import glob
import torch
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from utils.network_weight import UNet
from utils.network import UNet as HUNet
import argparse
from utils.draw_skeleton import create_colors, draw_skeleton
from utils.bmi_calcultator import create_output_directory, BMI_calculator
import tkinter as tk
from tkinter import filedialog


#python HWFinder.py -i [IMAGE ADDRESS] -g [GPU NUMBER] -r [RESOLUTION]

average_height = 0
average_weight = 0
images_count = 0

if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()

    image_file_paths = filedialog.askopenfilenames(title="Selecciona tus 3 images de frente", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    image_paths_front = root.tk.splitlist(image_file_paths)

    image_file_paths = filedialog.askopenfilenames(title="Selecciona tus 3 images de lado", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    image_paths_side = root.tk.splitlist(image_file_paths)
    
    image_file_paths = filedialog.askopenfilenames(title="Selecciona tus 3 images de espalda", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    image_paths_back = root.tk.splitlist(image_file_paths)

    np.random.seed(23)
    parser = argparse.ArgumentParser(description="Height and Weight Information from Unconstrained Images")

    parser.add_argument('-i', '--image', type=str, required=False, help='Image Directory')
    parser.add_argument('-g', '--gpu', type=int, default=4, help='GPU selection')
    parser.add_argument('-r', '--resolution', type=int, required=False, help='Resolution for Square Image')
    args = parser.parse_args()

    
    model_h = HUNet(128)
    pretrained_model_h = torch.load('models/model_ep_48.pth.tar', map_location=torch.device('cpu'))

    model_w = UNet(128, 32, 32)
    pretrained_model_w = torch.load('models/model_ep_37.pth.tar', map_location=torch.device('cpu'))
    
    model_h.load_state_dict(pretrained_model_h["state_dict"])
    model_w.load_state_dict(pretrained_model_w["state_dict"])
    
    if torch.cuda.is_available():
        model = model_w.cuda(args.gpu)
    else:
        model = model_w

    test_folder_name = "test 1"
    image_sets = [image_paths_front, image_paths_side, image_paths_back]
    image_set_names = ['front', 'side', 'back']

    # Define the name of the main test folder
    test_folder_name = input("Digite su nombre: ")

    image_sets = [image_paths_front, image_paths_side, image_paths_back]
    image_set_names = ['front', 'side', 'back']

    for image_set, image_set_name in zip(image_sets, image_set_names):
        if torch.cuda.is_available():
            model = model_w.cuda(args.gpu)
        else:
            model = model_w

        subdirectory_path = os.path.join("Salida", test_folder_name, image_set_name)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)
        
        for image in image_set:
            assert ".jpg" in image or ".png" in image or ".jpeg" in image, "Please use .jpg or .png format"
            
            print(image)
            RES = 128
            
            X = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype('float32')
            scale = RES / max(X.shape[:2])
            
            X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
            
            if X_scaled.shape[1] > X_scaled.shape[0]:
                p_a = (RES - X_scaled.shape[0])//2
                p_b = (RES - X_scaled.shape[0])-p_a
                X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
            elif X_scaled.shape[1] <= X_scaled.shape[0]:
                p_a = (RES - X_scaled.shape[1])//2
                p_b = (RES - X_scaled.shape[1])-p_a
                X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
            
            o_img = X.copy()
            X /= 255
            X = transforms.ToTensor()(X).unsqueeze(0)
                
            if torch.cuda.is_available():
                X = X.cuda()
            
            model.eval()
            with torch.no_grad():
                m_p, j_p, _, w_p = model(X)
            
            del model
            
            if torch.cuda.is_available():
                model = model_h.cuda(args.gpu)
            else:
                model = model_h
                
            model.eval()
            with torch.no_grad():
                _, _, h_p = model(X)
            
            fformat = '.png'
                
            if '.jpg' in image:
                fformat = '.jpg'
            elif '.jpeg' in image:
                fformat = '.jpeg'        
                
            mask_out = m_p.argmax(1).squeeze().cpu().numpy()
            joint_out = j_p.argmax(1).squeeze().cpu().numpy()
            pred_2 = j_p.squeeze().cpu().numpy()
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_out.astype('uint8'))
            colors = create_colors(30)
            img_sk = np.zeros((128,128,3))
            
            joint_pos = []
                
            for i in range(1, num_labels):
                p_res = np.expand_dims((labels==i).astype(int),0) * pred_2
                
                ct_ = 1
                positions = []

                for i in range(1,19):
                    positions.append(np.unravel_index(p_res[ct_,:,:].argmax(), p_res[ct_,:,:].shape))
                    ct_ += 1
                    
                joint_pos.append(positions)
            
            mask_out_RGB = np.concatenate([255*mask_out[:, :, np.newaxis],
                                        255*mask_out[:, :, np.newaxis],
                                        mask_out[:, :, np.newaxis],
                                        ], axis=-1)
            
            layer = cv2.addWeighted(o_img.astype('uint8'), 0.55, mask_out_RGB.astype('uint8'), 0.45, 0) 
            img_sk = draw_skeleton(layer/255, joint_pos, colors)

            out_name = image.split("/")[-1].replace(fformat, '.mask.png')
            out_name_j = image.split("/")[-1].replace(fformat, '.joint.png')
            out_name_sk = image.split("/")[-1].replace(fformat, '.skeleton.png')

            output_directory = subdirectory_path
            height = 100 * h_p.item()
            weight = 100 * w_p.item()
            BMI = weight / (height / 100) ** 2
            average_height = average_height + height
            average_weight = average_weight + weight
            images_count = images_count + 1


            with open(os.path.join(output_directory, image.split("/")[-1].replace(fformat, '.info.txt')), 'w') as out_file:
                out_file.write("Image: " + image)
                out_file.write("\nAltura: {:.1f} cm\nPeso: {:.1f} kg".format(height, weight))
                out_file.write("\nIMC: " + str(BMI))
                out_file.write("\nEstado: " + BMI_calculator(BMI))

            cv2.imwrite(os.path.join(output_directory, out_name), (255 * mask_out).astype('uint8'))
            plt.imsave(os.path.join(output_directory, out_name_j), joint_out, cmap='jet')
            plt.imsave(os.path.join(output_directory, out_name_sk), img_sk)

            del model


    average_height = average_height / images_count
    average_weight = average_weight / images_count

    real_height = int(input("\nDigite en cm, su altura: "))
    height_error_percentage = abs(real_height - average_height) / average_height * 100
    average_weight = average_weight * (average_height**2 / real_height**2)

    print("\nAltura estimada: {:.1f} cm\nPeso estimado: {:.1f} kg".format(average_height, average_weight))
    print("\nPorcentaje de error de altura: " + str(height_error_percentage) + "%")
    print("Resultados guardados en el directorio salida")
        
