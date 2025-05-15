using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;


namespace cSharp
{
    partial class MainForm
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
            btnLandmarks = new Button();
            btnFaceAlign = new Button();
            btnFaceSimilarity = new Button();
            btnFaceEval = new Button();
            SuspendLayout();
            // 
            // btnLandmarks
            // 
            btnLandmarks.Location = new Point(35, 40);
            btnLandmarks.Name = "btnLandmarks";
            btnLandmarks.Size = new Size(128, 39);
            btnLandmarks.TabIndex = 0;
            btnLandmarks.Text = "face Landmarks";
            btnLandmarks.UseVisualStyleBackColor = true;
            btnLandmarks.Click += btnLandmarks_Click;
            // 
            // btnFaceAlign
            // 
            btnFaceAlign.Location = new Point(35, 85);
            btnFaceAlign.Name = "btnFaceAlign";
            btnFaceAlign.Size = new Size(128, 36);
            btnFaceAlign.TabIndex = 1;
            btnFaceAlign.Text = "face Align";
            btnFaceAlign.UseVisualStyleBackColor = true;
            btnFaceAlign.Click += btnFaceAlign_Click;
            // 
            // btnFaceSimilarity
            // 
            btnFaceSimilarity.Location = new Point(35, 127);
            btnFaceSimilarity.Name = "btnFaceSimilarity";
            btnFaceSimilarity.Size = new Size(128, 36);
            btnFaceSimilarity.TabIndex = 2;
            btnFaceSimilarity.Text = "face Similarity";
            btnFaceSimilarity.UseVisualStyleBackColor = true;
            btnFaceSimilarity.Click += btnFaceSimilarity_Click;
            // 
            // btnFaceEval
            // 
            btnFaceEval.Location = new Point(35, 169);
            btnFaceEval.Name = "btnFaceEval";
            btnFaceEval.Size = new Size(162, 36);
            btnFaceEval.TabIndex = 3;
            btnFaceEval.Text = "face Eval photo check";
            btnFaceEval.UseVisualStyleBackColor = true;
            btnFaceEval.Click += btnFaceEval_Click;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(btnFaceEval);
            Controls.Add(btnFaceSimilarity);
            Controls.Add(btnFaceAlign);
            Controls.Add(btnLandmarks);
            Name = "MainForm";
            Text = "MainForm";
            ResumeLayout(false);
        }

        #endregion

        private Button btnLandmarks;
        private Button btnFaceAlign;
        private Button btnFaceSimilarity;
        private Button btnFaceEval;
    }
}