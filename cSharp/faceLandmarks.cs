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
    public partial class faceLandmarks : Form
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "http://localhost:8000";
        private string selectedImagePath;

        public faceLandmarks()
        {
            InitializeComponent();
            _httpClient = new HttpClient();
        }

        private void btnSelectImage_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    selectedImagePath = openFileDialog.FileName;
                    picOriginal.Image = Image.FromFile(selectedImagePath);
                    btnDetect.Enabled = true;
                    lblStatus.Text = "Image selected. Click 'Detect Face' to process.";
                }
            }
        }

        private async void btnDetect_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(selectedImagePath))
            {
                MessageBox.Show("Please select an image first.");
                return;
            }

            try
            {
                btnDetect.Enabled = false;
                lblStatus.Text = "Processing...";

                // Read and encode image
                byte[] imageBytes = File.ReadAllBytes(selectedImagePath);
                string base64Image = Convert.ToBase64String(imageBytes);

                // Prepare request data
                var requestData = new
                {
                    base64image = base64Image,
                    output_imgType = Path.GetExtension(selectedImagePath).TrimStart('.')
                };

                // Send request
                var content = new StringContent(
                    JsonSerializer.Serialize(requestData),
                    Encoding.UTF8,
                    "application/json"
                );

                var response = await _httpClient.PostAsync($"{BaseUrl}/detect_landmarks", content);
                response.EnsureSuccessStatusCode();

                var responseContent = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<FaceDetectionResult>(responseContent);

                // Display results
                if (result.resultcode == "SUCCESS")
                {
                    picFaceMesh.Image = Base64ToImage(result.base64image_face);
                    picFaceDot.Image = Base64ToImage(result.base64image_faceDot);
                    picFaceDLib.Image = Base64ToImage(result.base64image_faceDLib);
                    picFaceDLibNumbered.Image = Base64ToImage(result.base64image_faceDLib_68_number);
                    lblStatus.Text = result.resultmessage;
                }
                else
                {
                    MessageBox.Show($"Error: {result.resultmessage}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    lblStatus.Text = "Error occurred during face detection.";
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                lblStatus.Text = "Error occurred during face detection.";
            }
            finally
            {
                btnDetect.Enabled = true;
            }
        }

        private Image Base64ToImage(string base64String)
        {
            byte[] imageBytes = Convert.FromBase64String(base64String);
            using (MemoryStream ms = new MemoryStream(imageBytes))
            {
                return Image.FromStream(ms);
            }
        }
    }

    public class FaceDetectionResult
    {
        public string base64image { get; set; }
        public string base64image_face { get; set; }
        public string base64image_faceDot { get; set; }
        public string base64image_faceDLib { get; set; }
        public string base64image_faceDLib_68_number { get; set; }
        public string resultcode { get; set; }
        public string resultmessage { get; set; }
    }
}