using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using FaceDetectionClient;

namespace cSharp
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
        }

        private void btnLandmarks_Click(object sender, EventArgs e)
        {
            faceLandmarks frm = new faceLandmarks();
            frm.Show();
        }

        private void btnFaceAlign_Click(object sender, EventArgs e)
        {
            faceAlign frm = new faceAlign();
            frm.Show();
        }

        private void btnFaceSimilarity_Click(object sender, EventArgs e)
        {
            faceSimilarity frm = new faceSimilarity();
            frm.Show();
        }

        private void btnFaceEval_Click(object sender, EventArgs e)
        {
            faceEval frm = new faceEval();
            frm.Show();
        }
    }
}
