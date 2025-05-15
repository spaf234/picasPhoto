using System;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Windows.Forms;


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
    public partial class faceSimilarity : Form
    {
        private string base64Image1;
        private string base64Image2;
        private readonly HttpClient client;

        public faceSimilarity()
        {
            InitializeComponent();
            client = new HttpClient();
            client.BaseAddress = new Uri("http://localhost:8000/");
        }

        private void btnSelectImage1_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.gif;*.bmp";
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    string imagePath = openFileDialog.FileName;
                    pictureBox1.Image = Image.FromFile(imagePath);
                    base64Image1 = ConvertImageToBase64(imagePath);
                    txtFile1.Text = Path.GetFileName(imagePath);
                }
            }
        }

        private void btnSelectImage2_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.gif;*.bmp";
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    string imagePath = openFileDialog.FileName;
                    pictureBox2.Image = Image.FromFile(imagePath);
                    base64Image2 = ConvertImageToBase64(imagePath);
                    txtFile2.Text = Path.GetFileName(imagePath);
                }
            }
        }

        private string ConvertImageToBase64(string imagePath)
        {
            byte[] imageBytes = File.ReadAllBytes(imagePath);
            return Convert.ToBase64String(imageBytes);
        }

        private async void btnCompare_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(base64Image1) || string.IsNullOrEmpty(base64Image2))
            {
                MessageBox.Show("Please select both images first.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            try
            {
                var requestData = new
                {
                    base64image1 = base64Image1,
                    base64image2 = base64Image2,
                    filename1 = txtFile1.Text,
                    filename2 = txtFile2.Text
                };

                var content = new StringContent(
                    JsonSerializer.Serialize(requestData),
                    Encoding.UTF8,
                    "application/json");

                var response = await client.PostAsync("face-similarity", content);
                var jsonResponse = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<FaceSimilarityResponse>(jsonResponse);

                if (result.resultcode == "SUCCESS")
                {
                    lblSimilarity.Text = $"{result.similarity_score:F2}%";
                }
                else
                {
                    MessageBox.Show(result.resultmessage, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }

    public class FaceSimilarityResponse
    {
        public float similarity_score { get; set; }
        public string resultcode { get; set; }
        public string resultmessage { get; set; }
    }
} 