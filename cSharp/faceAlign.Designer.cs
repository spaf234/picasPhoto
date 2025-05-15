using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace FaceDetectionClient
{
    partial class faceAlign
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.btnSelectImage = new System.Windows.Forms.Button();
            this.originalPictureBox = new System.Windows.Forms.PictureBox();
            this.alignedPictureBox = new System.Windows.Forms.PictureBox();
            this.maskPictureBox = new System.Windows.Forms.PictureBox();
            this.colorPictureBox = new System.Windows.Forms.PictureBox();
            this.greenMaskPictureBox = new System.Windows.Forms.PictureBox();


            
            this.lblStatus = new System.Windows.Forms.Label();
            this.lblOriginal = new System.Windows.Forms.Label();
            this.lblAligned = new System.Windows.Forms.Label();
            this.lblMask = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.originalPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.alignedPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.maskPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.colorPictureBox)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.greenMaskPictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // btnSelectImage
            // 
            this.btnSelectImage.Location = new System.Drawing.Point(12, 12);
            this.btnSelectImage.Name = "btnSelectImage";
            this.btnSelectImage.Size = new System.Drawing.Size(120, 30);
            this.btnSelectImage.TabIndex = 0;
            this.btnSelectImage.Text = "Select Image";
            this.btnSelectImage.UseVisualStyleBackColor = true;
            this.btnSelectImage.Click += new System.EventHandler(this.btnSelectImage_Click);
            // 
            // originalPictureBox
            // 
            this.originalPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.originalPictureBox.Location = new System.Drawing.Point(12, 80);
            this.originalPictureBox.Name = "originalPictureBox";
            this.originalPictureBox.Size = new System.Drawing.Size(300, 300);
            this.originalPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.originalPictureBox.TabIndex = 1;
            this.originalPictureBox.TabStop = false;
            // 
            // alignedPictureBox
            // 
            this.alignedPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.alignedPictureBox.Location = new System.Drawing.Point(318, 80);
            this.alignedPictureBox.Name = "alignedPictureBox";
            this.alignedPictureBox.Size = new System.Drawing.Size(300, 300);
            this.alignedPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.alignedPictureBox.TabIndex = 2;
            this.alignedPictureBox.TabStop = false;
            // 
            // maskPictureBox
            // 
            this.maskPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.maskPictureBox.Location = new System.Drawing.Point(624, 80);
            this.maskPictureBox.Name = "maskPictureBox";
            this.maskPictureBox.Size = new System.Drawing.Size(300, 300);
            this.maskPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.maskPictureBox.TabIndex = 3;
            this.maskPictureBox.TabStop = false;

            //  
            // greenMaskPictureBox
            // 
            this.greenMaskPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.greenMaskPictureBox.Location = new System.Drawing.Point(930, 80);
            this.greenMaskPictureBox.Name = "greenMaskPictureBox";
            

            // 
            // colorPictureBox
            // 
            this.colorPictureBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.colorPictureBox.Location = new System.Drawing.Point(624, 70);
            this.colorPictureBox.Name = "maskPictureBox";
            this.colorPictureBox.Size = new System.Drawing.Size(300, 60);
            this.colorPictureBox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.colorPictureBox.TabIndex = 3;
            this.colorPictureBox.TabStop = false;
            // 
            // lblStatus
            // 
            this.lblStatus.AutoSize = true;
            this.lblStatus.Location = new System.Drawing.Point(138, 20);
            this.lblStatus.Name = "lblStatus";
            this.lblStatus.Size = new System.Drawing.Size(50, 15);
            this.lblStatus.TabIndex = 4;
            this.lblStatus.Text = "Status: ";
            // 
            // lblOriginal
            // 
            this.lblOriginal.AutoSize = true;
            this.lblOriginal.Location = new System.Drawing.Point(12, 62);
            this.lblOriginal.Name = "lblOriginal";
            this.lblOriginal.Size = new System.Drawing.Size(50, 15);
            this.lblOriginal.TabIndex = 5;
            this.lblOriginal.Text = "Original";
            // 
            // lblAligned
            // 
            this.lblAligned.AutoSize = true;
            this.lblAligned.Location = new System.Drawing.Point(318, 62);
            this.lblAligned.Name = "lblAligned";
            this.lblAligned.Size = new System.Drawing.Size(50, 15);
            this.lblAligned.TabIndex = 6;
            this.lblAligned.Text = "Aligned";
            // 
            // lblMask
            // 
            this.lblMask.AutoSize = true;
            this.lblMask.Location = new System.Drawing.Point(624, 62);
            this.lblMask.Name = "lblMask";
            this.lblMask.Size = new System.Drawing.Size(38, 15);
            this.lblMask.TabIndex = 7;
            this.lblMask.Text = "Mask";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(934, 391);
            this.Controls.Add(this.lblMask);
            this.Controls.Add(this.lblAligned);
            this.Controls.Add(this.lblOriginal);
            this.Controls.Add(this.lblStatus);
            this.Controls.Add(this.maskPictureBox);
            this.Controls.Add(this.alignedPictureBox);
            this.Controls.Add(this.originalPictureBox);
            this.Controls.Add(this.btnSelectImage);
            this.Name = "Form1";
            this.Text = "Face Alignment App";
            ((System.ComponentModel.ISupportInitialize)(this.originalPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.alignedPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.maskPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.colorPictureBox)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.greenMaskPictureBox)).EndInit();   
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion

        private System.Windows.Forms.Button btnSelectImage;
        private System.Windows.Forms.PictureBox originalPictureBox;
        private System.Windows.Forms.PictureBox alignedPictureBox;
        private System.Windows.Forms.PictureBox maskPictureBox;
        private System.Windows.Forms.PictureBox colorPictureBox;
        private System.Windows.Forms.PictureBox greenMaskPictureBox;




        
        private System.Windows.Forms.Label lblStatus;
        private System.Windows.Forms.Label lblOriginal;
        private System.Windows.Forms.Label lblAligned;
        private System.Windows.Forms.Label lblMask;
    }
} 