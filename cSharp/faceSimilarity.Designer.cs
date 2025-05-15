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
    partial class faceSimilarity
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
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.btnSelectImage1 = new System.Windows.Forms.Button();
            this.btnSelectImage2 = new System.Windows.Forms.Button();
            this.btnCompare = new System.Windows.Forms.Button();
            this.lblSimilarity = new System.Windows.Forms.Label();
            this.txtFile1 = new System.Windows.Forms.TextBox();
            this.txtFile2 = new System.Windows.Forms.TextBox();
            
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.SuspendLayout();
            
            // pictureBox1
            this.pictureBox1.Location = new System.Drawing.Point(12, 41);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(300, 300);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            
            // pictureBox2
            this.pictureBox2.Location = new System.Drawing.Point(488, 41);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(300, 300);
            this.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBox2.TabIndex = 1;
            this.pictureBox2.TabStop = false;
            
            // btnSelectImage1
            this.btnSelectImage1.Location = new System.Drawing.Point(12, 12);
            this.btnSelectImage1.Name = "btnSelectImage1";
            this.btnSelectImage1.Size = new System.Drawing.Size(100, 23);
            this.btnSelectImage1.TabIndex = 2;
            this.btnSelectImage1.Text = "Select Image 1";
            this.btnSelectImage1.Click += new System.EventHandler(this.btnSelectImage1_Click);
            
            // btnSelectImage2
            this.btnSelectImage2.Location = new System.Drawing.Point(488, 12);
            this.btnSelectImage2.Name = "btnSelectImage2";
            this.btnSelectImage2.Size = new System.Drawing.Size(100, 23);
            this.btnSelectImage2.TabIndex = 3;
            this.btnSelectImage2.Text = "Select Image 2";
            this.btnSelectImage2.Click += new System.EventHandler(this.btnSelectImage2_Click);
            
            // btnCompare
            this.btnCompare.Location = new System.Drawing.Point(350, 180);
            this.btnCompare.Name = "btnCompare";
            this.btnCompare.Size = new System.Drawing.Size(100, 23);
            this.btnCompare.TabIndex = 4;
            this.btnCompare.Text = "Compare";
            this.btnCompare.Click += new System.EventHandler(this.btnCompare_Click);
            
            // lblSimilarity
            this.lblSimilarity.AutoSize = true;
            this.lblSimilarity.Font = new System.Drawing.Font("Microsoft Sans Serif", 14F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSimilarity.Location = new System.Drawing.Point(347, 150);
            this.lblSimilarity.Name = "lblSimilarity";
            this.lblSimilarity.Size = new System.Drawing.Size(106, 24);
            this.lblSimilarity.TabIndex = 5;
            this.lblSimilarity.Text = "0.00%";
            this.lblSimilarity.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            
            // txtFile1
            this.txtFile1.Location = new System.Drawing.Point(118, 14);
            this.txtFile1.Name = "txtFile1";
            this.txtFile1.ReadOnly = true;
            this.txtFile1.Size = new System.Drawing.Size(194, 20);
            this.txtFile1.TabIndex = 6;
            
            // txtFile2
            this.txtFile2.Location = new System.Drawing.Point(594, 14);
            this.txtFile2.Name = "txtFile2";
            this.txtFile2.ReadOnly = true;
            this.txtFile2.Size = new System.Drawing.Size(194, 20);
            this.txtFile2.TabIndex = 7;
            
            // Form1
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 353);
            this.Controls.Add(this.txtFile2);
            this.Controls.Add(this.txtFile1);
            this.Controls.Add(this.lblSimilarity);
            this.Controls.Add(this.btnCompare);
            this.Controls.Add(this.btnSelectImage2);
            this.Controls.Add(this.btnSelectImage1);
            this.Controls.Add(this.pictureBox2);
            this.Controls.Add(this.pictureBox1);
            this.Name = "Form1";
            this.Text = "Face Similarity Comparison";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.PictureBox pictureBox2;
        private System.Windows.Forms.Button btnSelectImage1;
        private System.Windows.Forms.Button btnSelectImage2;
        private System.Windows.Forms.Button btnCompare;
        private System.Windows.Forms.Label lblSimilarity;
        private System.Windows.Forms.TextBox txtFile1;
        private System.Windows.Forms.TextBox txtFile2;
    }
} 