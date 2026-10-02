import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models

# -----------------------------------------------------------
# 1. Class Definitions (EMDS-6 Benchmark)
# -----------------------------------------------------------
CLASS_NAMES = [
    "Actinophrys", "Arcella", "Aspidisca", "Codosiga", "Colpoda",
    "Epistylis", "Euglypha", "Paramecium", "Rotifera", "Vorticella",
    "Noctiluca", "Ceratium", "Stentor", "Siprostomum", "Keratella Quadrala",
    "Euglena", "Gymnodinium", "Gonyaulax", "Phacus", "Stylongchia", "Synchaeta"
]

# -----------------------------------------------------------
# 2. Model Architecture Definitions
# -----------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = self.conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x1], dim=1))
        return self.outc(x)

def build_classifier(num_classes=21):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

# -----------------------------------------------------------
# 3. GUI Desktop Application (Three-Panel Layout)
# -----------------------------------------------------------
class MicroorganismApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Two-Stage Intelligent EM Analysis System (EMDS-6)")
        self.root.geometry("1040x660")
        self.root.resizable(False, False)

        # Device & preprocessing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_size = (224, 224)

        # Load models
        self.load_models()

        # Build interface
        self.setup_ui()

    def load_models(self):
        try:
            self.seg_net = UNet(in_channels=3, out_channels=1).to(self.device)
            self.seg_net.load_state_dict(torch.load("best_seg_unet.pth", map_location=self.device))
            self.seg_net.eval()

            self.cls_net = build_classifier(num_classes=len(CLASS_NAMES)).to(self.device)
            self.cls_net.load_state_dict(torch.load("best_cls_model.pth", map_location=self.device))
            self.cls_net.eval()
            print("Model weights loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load weights:\n{str(e)}")
            sys.exit(1)

    def setup_ui(self):
        # Top toolbar
        top_frame = tk.Frame(self.root, pady=12)
        top_frame.pack(fill="x")

        title_lbl = tk.Label(
            top_frame, 
            text="Environmental Microorganism Segmentation & Classification System", 
            font=("Segoe UI", 15, "bold")
        )
        title_lbl.pack(side="left", padx=20)

        self.btn_open = tk.Button(
            top_frame, 
            text=" Open Microscopic Image ", 
            font=("Segoe UI", 10, "bold"),
            bg="#1a73e8", 
            fg="white", 
            relief="flat", 
            command=self.process_image,
            cursor="hand2", 
            padx=12, 
            pady=5
        )
        self.btn_open.pack(side="right", padx=20)

        # Separator line
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=15)

        # Main workspace (three columns)
        main_frame = tk.Frame(self.root, pady=15)
        main_frame.pack(fill="both", expand=True, padx=20)

        # Column 1: Raw Input
        col1 = tk.LabelFrame(main_frame, text=" 1. Raw Microscopic Slide ", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        col1.grid(row=0, column=0, padx=10, sticky="nsew")

        self.lbl_orig = tk.Label(col1, text="Click 'Open Microscopic Image'\nto select an image", bg="#f8f9fa", width=32, height=16)
        self.lbl_orig.pack(pady=10)

        # Column 2: Stage 1 Segmentation Output
        col2 = tk.LabelFrame(main_frame, text=" 2. Stage 1 Segmented Foreground ", font=("Segoe UI", 11, "bold"), padx=10, pady=10)
        col2.grid(row=0, column=1, padx=10, sticky="nsew")

        self.lbl_mask = tk.Label(col2, text="Waiting for\nU-Net segmentation", bg="#f8f9fa", width=32, height=16)
        self.lbl_mask.pack(pady=10)

        # Column 3: Stage 2 Classification Results
        col3 = tk.LabelFrame(main_frame, text=" 3. Stage 2 Predictions & Confidence ", font=("Segoe UI", 11, "bold"), padx=15, pady=10)
        col3.grid(row=0, column=2, padx=10, sticky="nsew")

        # Result labels and progress bars
        self.top_labels = []
        self.top_bars = []
        for i in range(3):
            lbl = tk.Label(col3, text=f"Top {i+1}: --", font=("Segoe UI", 11, "bold" if i==0 else "normal"), anchor="w")
            lbl.pack(fill="x", pady=(10 if i==0 else 4, 2))
            bar = ttk.Progressbar(col3, length=240, mode='determinate')
            bar.pack(fill="x", pady=(0, 6))
            self.top_labels.append(lbl)
            self.top_bars.append(bar)

        self.tip_lbl = tk.Label(col3, text="", font=("Segoe UI", 9), fg="#666666", wraplength=250, justify="left")
        self.tip_lbl.pack(fill="x", pady=15)

        # Column grid weighting
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # Bottom status bar
        self.status_var = tk.StringVar(value=f"Device: {self.device.type.upper()} | System Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w", font=("Segoe UI", 9), padx=10)
        status_bar.pack(side="bottom", fill="x")

    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Microscopic Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
        )
        if not file_path:
            return

        try:
            self.status_var.set("Running two-stage cascaded inference...")
            self.root.update()

            # 1. Image loading and resizing
            raw_img = Image.open(file_path).convert("RGB")
            resized_img = raw_img.resize(self.img_size, Image.BILINEAR)

            # 2. Tensor conversion & normalization
            tensor_img = TF.to_tensor(resized_img)
            norm_tensor = self.norm(tensor_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Stage 1: Segmentation
                seg_logits = self.seg_net(norm_tensor)
                pred_mask = (torch.sigmoid(seg_logits) > 0.5).float()
                
                # Masked foreground input for Stage 2
                masked_tensor = norm_tensor * pred_mask
                cls_logits = self.cls_net(masked_tensor)
                probs = torch.softmax(cls_logits[0], dim=0)

            # 3. Update panel 1: Raw image
            disp_orig = raw_img.copy()
            disp_orig.thumbnail((260, 260))
            tk_orig = ImageTk.PhotoImage(disp_orig)
            self.lbl_orig.config(image=tk_orig, text="", width=260, height=260)
            self.lbl_orig.image = tk_orig

            # 4. Update panel 2: Segmented foreground (RGB element-wise multiply Mask)
            mask_np = pred_mask.squeeze().cpu().numpy()
            orig_np = np.array(resized_img)
            masked_np = (orig_np * mask_np[:, :, None]).astype(np.uint8)
            disp_masked = Image.fromarray(masked_np)
            disp_masked.thumbnail((260, 260))
            tk_masked = ImageTk.PhotoImage(disp_masked)
            self.lbl_mask.config(image=tk_masked, text="", width=260, height=260)
            self.lbl_mask.image = tk_masked

            # 5. Update panel 3: Top-3 predictions and confidence bars
            top3_probs, top3_indices = torch.topk(probs, 3)
            for i in range(3):
                cls_name = CLASS_NAMES[top3_indices[i].item()]
                prob_val = top3_probs[i].item() * 100
                self.top_labels[i].config(text=f"Top {i+1}: {cls_name} ({prob_val:.2f}%)")
                self.top_bars[i]['value'] = prob_val

            self.tip_lbl.config(
                text=f"File: {os.path.basename(file_path)}\nStage 1: Background noise removed.\nStage 2: Features aligned for classification."
            )
            self.status_var.set(f"Completed | Predicted: {CLASS_NAMES[top3_indices[0].item()]}")

        except Exception as e:
            messagebox.showerror("Inference Error", f"An error occurred during processing:\n{str(e)}")
            self.status_var.set("Inference Failed")

# -----------------------------------------------------------
# 4. Main Entry Point
# -----------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = MicroorganismApp(root)
    root.mainloop()