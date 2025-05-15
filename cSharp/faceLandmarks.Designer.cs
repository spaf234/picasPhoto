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
    partial class faceLandmarks
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
            btnSelectImage = new Button();
            btnDetect = new Button();
            picOriginal = new PictureBox();
            picFaceMesh = new PictureBox();
            picFaceDot = new PictureBox();
            picFaceDLib = new PictureBox();
            picFaceDLibNumbered = new PictureBox();
            lblStatus = new Label();
            groupBox1 = new GroupBox();
            groupBox2 = new GroupBox();
            groupBox3 = new GroupBox();
            groupBox4 = new GroupBox();
            groupBox5 = new GroupBox();
            groupBox6 = new GroupBox();
            ((System.ComponentModel.ISupportInitialize)picOriginal).BeginInit();
            ((System.ComponentModel.ISupportInitialize)picFaceMesh).BeginInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDot).BeginInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDLib).BeginInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDLibNumbered).BeginInit();
            groupBox1.SuspendLayout();
            groupBox2.SuspendLayout();
            groupBox3.SuspendLayout();
            groupBox4.SuspendLayout();
            groupBox5.SuspendLayout();
            SuspendLayout();
            // 
            // btnSelectImage
            // 
            btnSelectImage.Location = new Point(10, 11);
            btnSelectImage.Name = "btnSelectImage";
            btnSelectImage.Size = new Size(105, 28);
            btnSelectImage.TabIndex = 0;
            btnSelectImage.Text = "Select Image";
            btnSelectImage.UseVisualStyleBackColor = true;
            btnSelectImage.Click += btnSelectImage_Click;
            // 
            // btnDetect
            // 
            btnDetect.Enabled = false;
            btnDetect.Location = new Point(124, 11);
            btnDetect.Name = "btnDetect";
            btnDetect.Size = new Size(105, 28);
            btnDetect.TabIndex = 1;
            btnDetect.Text = "Detect Face";
            btnDetect.UseVisualStyleBackColor = true;
            btnDetect.Click += btnDetect_Click;
            // 
            // picOriginal
            // 
            picOriginal.BorderStyle = BorderStyle.FixedSingle;
            picOriginal.Dock = DockStyle.Fill;
            picOriginal.Location = new Point(3, 19);
            picOriginal.Name = "picOriginal";
            picOriginal.Size = new Size(256, 259);
            picOriginal.SizeMode = PictureBoxSizeMode.Zoom;
            picOriginal.TabIndex = 2;
            picOriginal.TabStop = false;
            // 
            // picFaceMesh
            // 
            picFaceMesh.BorderStyle = BorderStyle.FixedSingle;
            picFaceMesh.Dock = DockStyle.Fill;
            picFaceMesh.Location = new Point(3, 19);
            picFaceMesh.Name = "picFaceMesh";
            picFaceMesh.Size = new Size(256, 259);
            picFaceMesh.SizeMode = PictureBoxSizeMode.Zoom;
            picFaceMesh.TabIndex = 3;
            picFaceMesh.TabStop = false;
            // 
            // picFaceDot
            // 
            picFaceDot.BorderStyle = BorderStyle.FixedSingle;
            picFaceDot.Dock = DockStyle.Fill;
            picFaceDot.Location = new Point(3, 19);
            picFaceDot.Name = "picFaceDot";
            picFaceDot.Size = new Size(256, 259);
            picFaceDot.SizeMode = PictureBoxSizeMode.Zoom;
            picFaceDot.TabIndex = 4;
            picFaceDot.TabStop = false;
            // 
            // picFaceDLib
            // 
            picFaceDLib.BorderStyle = BorderStyle.FixedSingle;
            picFaceDLib.Dock = DockStyle.Fill;
            picFaceDLib.Location = new Point(3, 19);
            picFaceDLib.Name = "picFaceDLib";
            picFaceDLib.Size = new Size(256, 259);
            picFaceDLib.SizeMode = PictureBoxSizeMode.Zoom;
            picFaceDLib.TabIndex = 5;
            picFaceDLib.TabStop = false;
            // 
            // picFaceDLibNumbered
            // 
            picFaceDLibNumbered.BorderStyle = BorderStyle.FixedSingle;
            picFaceDLibNumbered.Dock = DockStyle.Fill;
            picFaceDLibNumbered.Location = new Point(3, 300);
            picFaceDLibNumbered.Name = "picFaceDLibNumbered";
            picFaceDLibNumbered.Size = new Size(512, 500);
            picFaceDLibNumbered.SizeMode = PictureBoxSizeMode.Zoom;
            picFaceDLibNumbered.TabIndex = 50;
            picFaceDLibNumbered.TabStop = false;
            // 
            // lblStatus
            // 
            lblStatus.AutoSize = true;
            lblStatus.Location = new Point(10, 47);
            lblStatus.Name = "lblStatus";
            lblStatus.Size = new Size(0, 15);
            lblStatus.TabIndex = 6;
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(picOriginal);
            groupBox1.Location = new Point(10, 75);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(262, 281);
            groupBox1.TabIndex = 7;
            groupBox1.TabStop = false;
            groupBox1.Text = "Original Image";
            // 
            // groupBox2
            // 
            groupBox2.Controls.Add(picFaceMesh);
            groupBox2.Location = new Point(282, 75);
            groupBox2.Name = "groupBox2";
            groupBox2.Size = new Size(262, 281);
            groupBox2.TabIndex = 8;
            groupBox2.TabStop = false;
            groupBox2.Text = "Face Mesh";
            // 
            // groupBox3
            // 
            groupBox3.Controls.Add(picFaceDot);
            groupBox3.Location = new Point(553, 75);
            groupBox3.Name = "groupBox3";
            groupBox3.Size = new Size(262, 281);
            groupBox3.TabIndex = 9;
            groupBox3.TabStop = false;
            groupBox3.Text = "Face Dot";
            // 
            // groupBox4
            // 
            groupBox4.Controls.Add(picFaceDLib);
            groupBox4.Location = new Point(824, 75);
            groupBox4.Name = "groupBox4";
            groupBox4.Size = new Size(262, 281);
            groupBox4.TabIndex = 10;
            groupBox4.TabStop = false;
            groupBox4.Text = "Face DLib";
            // 
            // groupBox5
            // 
            
            groupBox5.Location = new Point(10, 175);
            groupBox5.Name = "groupBox5";
            groupBox5.Size = new Size(500, 650);
            groupBox5.TabIndex = 10;
            groupBox5.TabStop = false;
            groupBox5.Text = "Face DLib Numbered";
            // 
            // groupBox6
            // 
            groupBox6.Controls.Add(picFaceDLibNumbered);
            groupBox6.Location = new Point(31, 395);
            groupBox6.Name = "groupBox6";
            groupBox6.Size = new Size(636, 748);
            groupBox6.TabIndex = 11;
            groupBox6.TabStop = false;
            groupBox6.Text = "groupBox6";
            // 
            // faceLandmarks
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1350, 1185);
            Controls.Add(groupBox6);
            Controls.Add(groupBox4);
            Controls.Add(groupBox3);
            Controls.Add(groupBox2);
            Controls.Add(groupBox1);
            Controls.Add(lblStatus);
            Controls.Add(btnDetect);
            Controls.Add(btnSelectImage);
            Name = "faceLandmarks";
            Text = "Face Detection Client";
            ((System.ComponentModel.ISupportInitialize)picOriginal).EndInit();
            ((System.ComponentModel.ISupportInitialize)picFaceMesh).EndInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDot).EndInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDLib).EndInit();
            ((System.ComponentModel.ISupportInitialize)picFaceDLibNumbered).EndInit();
            groupBox1.ResumeLayout(false);
            groupBox2.ResumeLayout(false);
            groupBox3.ResumeLayout(false);
            groupBox4.ResumeLayout(false);
            groupBox5.ResumeLayout(false);
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private System.Windows.Forms.Button btnSelectImage;
        private System.Windows.Forms.Button btnDetect;
        private System.Windows.Forms.PictureBox picOriginal;
        private System.Windows.Forms.PictureBox picFaceMesh;
        private System.Windows.Forms.PictureBox picFaceDot;
        private System.Windows.Forms.PictureBox picFaceDLib;
        private System.Windows.Forms.PictureBox picFaceDLibNumbered;



        
        private System.Windows.Forms.Label lblStatus;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.GroupBox groupBox5;
        private GroupBox groupBox6;
    }
} 