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
    partial class faceEval
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
            pictureBoxOriginal = new PictureBox();
            pictureBoxFaceDetect = new PictureBox();
            pictureBoxHeadDetect = new PictureBox();
            pictureBoxBgRemoved = new PictureBox();
            pictureBoxBgRemovedCanny = new PictureBox();
            pictureBoxBgRemovedCrownUpd = new PictureBox();
            openFileDialog = new OpenFileDialog();
            btnCheck = new Button();
            dataGridView = new DataGridView();
            Column1 = new DataGridViewTextBoxColumn();
            Column2 = new DataGridViewTextBoxColumn();
            Column3 = new DataGridViewTextBoxColumn();
            groupBox1_org = new GroupBox();
            groupBox2_bgRemove = new GroupBox();
            groupBox3_bgRemove_Canny = new GroupBox();
            groupBox4_face_detect = new GroupBox();
            groupBox6_head_detect = new GroupBox();
            groupBox5_bgRemove_crown_upd = new GroupBox();
            groupBox1 = new GroupBox();
            pictureBox1 = new PictureBox();
            ((System.ComponentModel.ISupportInitialize)pictureBoxOriginal).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxFaceDetect).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxHeadDetect).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemoved).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemovedCanny).BeginInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemovedCrownUpd).BeginInit();
            ((System.ComponentModel.ISupportInitialize)dataGridView).BeginInit();
            groupBox1_org.SuspendLayout();
            groupBox2_bgRemove.SuspendLayout();
            groupBox3_bgRemove_Canny.SuspendLayout();
            groupBox4_face_detect.SuspendLayout();
            groupBox6_head_detect.SuspendLayout();
            groupBox5_bgRemove_crown_upd.SuspendLayout();
            groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            SuspendLayout();
            // 
            // btnSelectImage
            // 
            btnSelectImage.Location = new Point(14, 14);
            btnSelectImage.Margin = new Padding(4, 3, 4, 3);
            btnSelectImage.Name = "btnSelectImage";
            btnSelectImage.Size = new Size(140, 35);
            btnSelectImage.TabIndex = 0;
            btnSelectImage.Text = "Select Image";
            btnSelectImage.UseVisualStyleBackColor = true;
            btnSelectImage.Click += BtnSelectImage_Click;
            // 
            // pictureBoxOriginal
            // 
            pictureBoxOriginal.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxOriginal.Dock = DockStyle.Fill;
            pictureBoxOriginal.Location = new Point(3, 19);
            pictureBoxOriginal.Margin = new Padding(4, 3, 4, 3);
            pictureBoxOriginal.Name = "pictureBoxOriginal";
            pictureBoxOriginal.Size = new Size(230, 280);
            pictureBoxOriginal.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxOriginal.TabIndex = 1;
            pictureBoxOriginal.TabStop = false;
            // 
            // pictureBoxFaceDetect
            // 
            pictureBoxFaceDetect.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxFaceDetect.Dock = DockStyle.Fill;
            pictureBoxFaceDetect.Location = new Point(3, 19);
            pictureBoxFaceDetect.Margin = new Padding(4, 3, 4, 3);
            pictureBoxFaceDetect.Name = "pictureBoxFaceDetect";
            pictureBoxFaceDetect.Size = new Size(232, 268);
            pictureBoxFaceDetect.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxFaceDetect.TabIndex = 2;
            pictureBoxFaceDetect.TabStop = false;
            // 
            // pictureBoxHeadDetect
            // 
            pictureBoxHeadDetect.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxHeadDetect.Dock = DockStyle.Fill;
            pictureBoxHeadDetect.Location = new Point(3, 19);
            pictureBoxHeadDetect.Margin = new Padding(4, 3, 4, 3);
            pictureBoxHeadDetect.Name = "pictureBoxHeadDetect";
            pictureBoxHeadDetect.Size = new Size(217, 265);
            pictureBoxHeadDetect.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxHeadDetect.TabIndex = 3;
            pictureBoxHeadDetect.TabStop = false;
            // 
            // pictureBoxBgRemoved
            // 
            pictureBoxBgRemoved.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxBgRemoved.Dock = DockStyle.Fill;
            pictureBoxBgRemoved.Location = new Point(3, 19);
            pictureBoxBgRemoved.Margin = new Padding(4, 3, 4, 3);
            pictureBoxBgRemoved.Name = "pictureBoxBgRemoved";
            pictureBoxBgRemoved.Size = new Size(232, 268);
            pictureBoxBgRemoved.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxBgRemoved.TabIndex = 4;
            pictureBoxBgRemoved.TabStop = false;
            // 
            // pictureBoxBgRemovedCanny
            // 
            pictureBoxBgRemovedCanny.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxBgRemovedCanny.Dock = DockStyle.Fill;
            pictureBoxBgRemovedCanny.Location = new Point(3, 19);
            pictureBoxBgRemovedCanny.Margin = new Padding(4, 3, 4, 3);
            pictureBoxBgRemovedCanny.Name = "pictureBoxBgRemovedCanny";
            pictureBoxBgRemovedCanny.Size = new Size(232, 268);
            pictureBoxBgRemovedCanny.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxBgRemovedCanny.TabIndex = 5;
            pictureBoxBgRemovedCanny.TabStop = false;
            // 
            // pictureBoxBgRemovedCrownUpd
            // 
            pictureBoxBgRemovedCrownUpd.BorderStyle = BorderStyle.FixedSingle;
            pictureBoxBgRemovedCrownUpd.Dock = DockStyle.Fill;
            pictureBoxBgRemovedCrownUpd.Location = new Point(3, 19);
            pictureBoxBgRemovedCrownUpd.Margin = new Padding(4, 3, 4, 3);
            pictureBoxBgRemovedCrownUpd.Name = "pictureBoxBgRemovedCrownUpd";
            pictureBoxBgRemovedCrownUpd.Size = new Size(232, 268);
            pictureBoxBgRemovedCrownUpd.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxBgRemovedCrownUpd.TabIndex = 6;
            pictureBoxBgRemovedCrownUpd.TabStop = false;
            // 
            // openFileDialog
            // 
            openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
            openFileDialog.Title = "Select Passport Photo";
            // 
            // btnCheck
            // 
            btnCheck.Location = new Point(161, 14);
            btnCheck.Margin = new Padding(4, 3, 4, 3);
            btnCheck.Name = "btnCheck";
            btnCheck.Size = new Size(140, 35);
            btnCheck.TabIndex = 7;
            btnCheck.Text = "Check Photo";
            btnCheck.UseVisualStyleBackColor = true;
            btnCheck.Click += BtnCheck_Click;
            // 
            // dataGridView
            // 
            dataGridView.AllowUserToAddRows = false;
            dataGridView.AllowUserToDeleteRows = false;
            dataGridView.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridView.Columns.AddRange(new DataGridViewColumn[] { Column1, Column2, Column3 });
            dataGridView.Location = new Point(971, 180);
            dataGridView.Margin = new Padding(4, 3, 4, 3);
            dataGridView.Name = "dataGridView";
            dataGridView.ReadOnly = true;
            dataGridView.RowHeadersVisible = false;
            dataGridView.Size = new Size(321, 473);
            dataGridView.TabIndex = 8;
            // 
            // Column1
            // 
            Column1.HeaderText = "Metric";
            Column1.Name = "Column1";
            Column1.ReadOnly = true;
            Column1.Width = 150;
            // 
            // Column2
            // 
            Column2.HeaderText = "Score";
            Column2.Name = "Column2";
            Column2.ReadOnly = true;
            Column2.Width = 80;
            // 
            // Column3
            // 
            Column3.HeaderText = "Pass/Fail";
            Column3.Name = "Column3";
            Column3.ReadOnly = true;
            Column3.Width = 80;
            // 
            // groupBox1_org
            // 
            groupBox1_org.Controls.Add(pictureBoxOriginal);
            groupBox1_org.Location = new Point(16, 55);
            groupBox1_org.Name = "groupBox1_org";
            groupBox1_org.Size = new Size(236, 302);
            groupBox1_org.TabIndex = 9;
            groupBox1_org.TabStop = false;
            groupBox1_org.Text = "groupBox1_org";
            // 
            // groupBox2_bgRemove
            // 
            groupBox2_bgRemove.Controls.Add(pictureBoxBgRemoved);
            groupBox2_bgRemove.Location = new Point(329, 58);
            groupBox2_bgRemove.Name = "groupBox2_bgRemove";
            groupBox2_bgRemove.Size = new Size(238, 290);
            groupBox2_bgRemove.TabIndex = 10;
            groupBox2_bgRemove.TabStop = false;
            groupBox2_bgRemove.Text = "groupBox2_bgRemove";
            // 
            // groupBox3_bgRemove_Canny
            // 
            groupBox3_bgRemove_Canny.Controls.Add(pictureBoxBgRemovedCanny);
            groupBox3_bgRemove_Canny.Location = new Point(603, 55);
            groupBox3_bgRemove_Canny.Name = "groupBox3_bgRemove_Canny";
            groupBox3_bgRemove_Canny.Size = new Size(238, 290);
            groupBox3_bgRemove_Canny.TabIndex = 11;
            groupBox3_bgRemove_Canny.TabStop = false;
            groupBox3_bgRemove_Canny.Text = "groupBox3_bgRemove_Canny";
            // 
            // groupBox4_face_detect
            // 
            groupBox4_face_detect.Controls.Add(pictureBoxFaceDetect);
            groupBox4_face_detect.Location = new Point(227, 363);
            groupBox4_face_detect.Name = "groupBox4_face_detect";
            groupBox4_face_detect.Size = new Size(238, 290);
            groupBox4_face_detect.TabIndex = 12;
            groupBox4_face_detect.TabStop = false;
            groupBox4_face_detect.Text = "groupBox4_face_detect";
            // 
            // groupBox6_head_detect
            // 
            groupBox6_head_detect.Controls.Add(pictureBoxHeadDetect);
            groupBox6_head_detect.Location = new Point(743, 363);
            groupBox6_head_detect.Name = "groupBox6_head_detect";
            groupBox6_head_detect.Size = new Size(223, 287);
            groupBox6_head_detect.TabIndex = 13;
            groupBox6_head_detect.TabStop = false;
            groupBox6_head_detect.Text = "groupBox6_head_detect";
            // 
            // groupBox5_bgRemove_crown_upd
            // 
            groupBox5_bgRemove_crown_upd.Controls.Add(pictureBoxBgRemovedCrownUpd);
            groupBox5_bgRemove_crown_upd.Location = new Point(484, 363);
            groupBox5_bgRemove_crown_upd.Name = "groupBox5_bgRemove_crown_upd";
            groupBox5_bgRemove_crown_upd.Size = new Size(238, 290);
            groupBox5_bgRemove_crown_upd.TabIndex = 14;
            groupBox5_bgRemove_crown_upd.TabStop = false;
            groupBox5_bgRemove_crown_upd.Text = "groupBox5_bgRemove_crown_upd";
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(pictureBox1);
            groupBox1.Location = new Point(67, 440);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(87, 133);
            groupBox1.TabIndex = 15;
            groupBox1.TabStop = false;
            groupBox1.Text = "groupBox1";
            // 
            // pictureBox1
            // 
            pictureBox1.Dock = DockStyle.Fill;
            pictureBox1.Location = new Point(3, 19);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(81, 111);
            pictureBox1.TabIndex = 0;
            pictureBox1.TabStop = false;
            // 
            // faceEval
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1313, 1019);
            Controls.Add(groupBox1);
            Controls.Add(groupBox5_bgRemove_crown_upd);
            Controls.Add(groupBox6_head_detect);
            Controls.Add(groupBox4_face_detect);
            Controls.Add(groupBox3_bgRemove_Canny);
            Controls.Add(groupBox2_bgRemove);
            Controls.Add(groupBox1_org);
            Controls.Add(dataGridView);
            Controls.Add(btnCheck);
            Controls.Add(btnSelectImage);
            Margin = new Padding(4, 3, 4, 3);
            Name = "faceEval";
            Text = "Passport Photo Checker";
            ((System.ComponentModel.ISupportInitialize)pictureBoxOriginal).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxFaceDetect).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxHeadDetect).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemoved).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemovedCanny).EndInit();
            ((System.ComponentModel.ISupportInitialize)pictureBoxBgRemovedCrownUpd).EndInit();
            ((System.ComponentModel.ISupportInitialize)dataGridView).EndInit();
            groupBox1_org.ResumeLayout(false);
            groupBox2_bgRemove.ResumeLayout(false);
            groupBox3_bgRemove_Canny.ResumeLayout(false);
            groupBox4_face_detect.ResumeLayout(false);
            groupBox6_head_detect.ResumeLayout(false);
            groupBox5_bgRemove_crown_upd.ResumeLayout(false);
            groupBox1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            ResumeLayout(false);
        }

        #endregion

        private System.Windows.Forms.Button btnSelectImage;
        private System.Windows.Forms.PictureBox pictureBoxOriginal;
        private System.Windows.Forms.PictureBox pictureBoxFaceDetect;
        private System.Windows.Forms.PictureBox pictureBoxHeadDetect;
        private System.Windows.Forms.PictureBox pictureBoxBgRemoved;
        private System.Windows.Forms.PictureBox pictureBoxBgRemovedCanny;
        private System.Windows.Forms.PictureBox pictureBoxBgRemovedCrownUpd;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.Windows.Forms.Button btnCheck;
        private System.Windows.Forms.DataGridView dataGridView;
        private System.Windows.Forms.DataGridViewTextBoxColumn Column1;
        private System.Windows.Forms.DataGridViewTextBoxColumn Column2;
        private System.Windows.Forms.DataGridViewTextBoxColumn Column3;
        private GroupBox groupBox1_org;
        private GroupBox groupBox2_bgRemove;
        private GroupBox groupBox3_bgRemove_Canny;
        private GroupBox groupBox4_face_detect;
        private GroupBox groupBox6_head_detect;
        private GroupBox groupBox5_bgRemove_crown_upd;
        private GroupBox groupBox1;
        private PictureBox pictureBox1;
    }
}