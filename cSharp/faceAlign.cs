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
    public partial class faceAlign : Form
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "http://localhost:8000";

        public faceAlign()
        {
            InitializeComponent();
            _httpClient = new HttpClient();
        }

        private async void btnSelectImage_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog openFileDialog = new OpenFileDialog())
            {
                openFileDialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
                openFileDialog.Title = "Select an Image File";

                if (openFileDialog.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        // Display original image
                        originalPictureBox.Image = Image.FromFile(openFileDialog.FileName);
                        
                        // Convert image to base64
                        byte[] imageBytes = File.ReadAllBytes(openFileDialog.FileName);
                        string base64Image = Convert.ToBase64String(imageBytes);

                        // Prepare request data
                        var requestData = new
                        {
                            base64image = base64Image,
                            output_imgType = Path.GetExtension(openFileDialog.FileName).TrimStart('.')
                        };

                        // Send request to API
                        var content = new StringContent(
                            JsonSerializer.Serialize(requestData),
                            Encoding.UTF8,
                            "application/json");

                        var response = await _httpClient.PostAsync($"{BaseUrl}/align_face", content);
                        response.EnsureSuccessStatusCode();

                        // Parse response
                        var responseContent = await response.Content.ReadAsStringAsync();
                        var result = JsonSerializer.Deserialize<FaceAlignResponse>(responseContent);

                        // Display aligned image
                        if (!string.IsNullOrEmpty(result.base64image_align))
                        {
                            byte[] alignedImageBytes = Convert.FromBase64String(result.base64image_align);
                            using (MemoryStream ms = new MemoryStream(alignedImageBytes))
                            {
                                alignedPictureBox.Image = Image.FromStream(ms);
                            }
                        }

                        // Display background mask
                        if (!string.IsNullOrEmpty(result.base64image_bg_mask))
                        {
                            byte[] maskImageBytes = Convert.FromBase64String(result.base64image_bg_mask);
                            using (MemoryStream ms = new MemoryStream(maskImageBytes))
                            {
                                maskPictureBox.Image = Image.FromStream(ms);
                            }
                        }

                        // Display green background mask
                        if (!string.IsNullOrEmpty(result.base64image_bg_mask_green))
                        {
                            byte[] greenMaskImageBytes = Convert.FromBase64String(result.base64image_bg_mask_green);
                            using (MemoryStream ms = new MemoryStream(greenMaskImageBytes))
                            {
                                greenMaskPictureBox.Image = Image.FromStream(ms);
                            }
                        }

                        // Display background color
                        if (result.background_color != null && result.background_color.Length == 3)
                        {
                            // Create a bitmap with the background color
                            Bitmap colorBitmap = new Bitmap(100, 100);
                            using (Graphics g = Graphics.FromImage(colorBitmap))
                            {
                                Color bgColor = Color.FromArgb(result.background_color[0], 
                                                             result.background_color[1], 
                                                             result.background_color[2]);
                                g.Clear(bgColor);
                            }
                            colorPictureBox.Image = colorBitmap;
                        }

                        // Update status
                        lblStatus.Text = $"Status: {result.resultcode} - {result.resultmessage}";
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }
    }

    public class FaceAlignResponse
    {
        public string base64image_align { get; set; }
        public string base64image_bg_mask { get; set; }
        public string base64image_bg_mask_green { get; set; }
        public int[] background_color { get; set; }
        public string resultcode { get; set; }
        public string resultmessage { get; set; }
    }
}