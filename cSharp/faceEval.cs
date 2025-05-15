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
    public partial class faceEval : Form
    {
        private readonly HttpClient _httpClient;
        private const string BaseUrl = "http://localhost:8000";
        private string _selectedImagePath;
        private PictureBox[] pictureBoxes;

        public faceEval()
        {
            InitializeComponent();
            _httpClient = new HttpClient();

            // Initialize array of PictureBoxes
            pictureBoxes = new PictureBox[5];
            for (int i = 0; i < 5; i++)
            {
                pictureBoxes[i] = new PictureBox();
                pictureBoxes[i].SizeMode = PictureBoxSizeMode.Zoom;
                pictureBoxes[i].Width = 200;
                pictureBoxes[i].Height = 200;
                pictureBoxes[i].Location = new Point(10 + (i * 210), 300);
                this.Controls.Add(pictureBoxes[i]);
            }

            //btnSelectImage.Click += BtnSelectImage_Click;
            //btnCheck.Click += BtnCheck_Click;
        }

        private void BtnSelectImage_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                _selectedImagePath = openFileDialog.FileName;
                pictureBoxOriginal.Image = Image.FromFile(_selectedImagePath);
            }
        }

        private async void BtnCheck_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedImagePath))
            {
                MessageBox.Show("Please select an image first.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            try
            {
                // Convert image to base64
                byte[] imageBytes = File.ReadAllBytes(_selectedImagePath);
                string base64Image = Convert.ToBase64String(imageBytes);

                // Prepare request data
                var requestData = new
                {
                    base64image = base64Image,
                    output_imgType = Path.GetExtension(_selectedImagePath).TrimStart('.')
                };

                // Send request to API
                var content = new StringContent(
                    JsonSerializer.Serialize(requestData),
                    Encoding.UTF8,
                    "application/json");

                var response = await _httpClient.PostAsync($"{BaseUrl}/check_pspt_photo_quality", content);
                response.EnsureSuccessStatusCode();

                // Parse response
                var responseContent = await response.Content.ReadAsStringAsync();
                var result = JsonSerializer.Deserialize<PassportPhotoResult>(responseContent);

                // Update DataGridView
                UpdateResultsGrid(result);

                // Display processed images
                DisplayProcessedImage(result.base64image_bg_removed, pictureBoxBgRemoved);
                DisplayProcessedImage(result.base64image_bg_removed_canny, pictureBoxBgRemovedCanny);
                DisplayProcessedImage(result.base64image_face_detect, pictureBoxFaceDetect);
                DisplayProcessedImage(result.base64image_head_detect, pictureBoxHeadDetect);
                DisplayProcessedImage(result.base64image_bg_removed_canny_upd_crown, pictureBoxBgRemovedCrownUpd);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error checking passport photo: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void DisplayProcessedImage(string base64Image, PictureBox pictureBox)
        {
            if (!string.IsNullOrEmpty(base64Image))
            {
                byte[] imageBytes = Convert.FromBase64String(base64Image);
                using (var ms = new MemoryStream(imageBytes))
                {
                    pictureBox.Image = Image.FromStream(ms);
                }
            }
        }

        private void UpdateResultsGrid(PassportPhotoResult result)
        {
            dataGridView.Rows.Clear();

            AddResultRow("Face Height Proportion", result.face_height_proportion, result.face_height_proportion >= 0.3);
            AddResultRow("Face Width Proportion", result.face_width_proportion, result.face_width_proportion >= 0.3);
            AddResultRow("Head Height Proportion", result.head_height_proportion, result.head_height_proportion >= 0.3);
            AddResultRow("Head Width Proportion", result.head_width_proportion, result.head_width_proportion >= 0.3);
            AddResultRow("Blur", result.blur_score, result.blur_score >= 70);
            AddResultRow("Pixelation", result.pixelation_score, result.pixelation_score >= 70);
            AddResultRow("White Noise", result.white_noise_score, result.white_noise_score >= 70);
            AddResultRow("Contrast", result.contrast_score, result.contrast_score >= 70);
            AddResultRow("General Illumination", result.general_illumination_score, result.general_illumination_score >= 70);
            AddResultRow("Face Position", result.face_position_score, result.face_position_score >= 70);
            AddResultRow("Face Pose", result.face_pose_score, result.face_pose_score >= 70);
            AddResultRow("Expression", result.expression_score, result.expression_score >= 70);
            AddResultRow("Eyes Open", result.eyes_open_score, result.eyes_open_score >= 70);
            AddResultRow("Eyes Direction", result.eyes_direction_score, result.eyes_direction_score >= 70);
            AddResultRow("Mouth Open", result.mouth_open_score, result.mouth_open_score >= 70);
            AddResultRow("Hair Over Face", result.hair_over_face_score, result.hair_over_face_score >= 70);
            AddResultRow("Sunglasses", result.sunglasses_score, result.sunglasses_score >= 70);
            AddResultRow("Glasses Reflection", result.glasses_reflection_score, result.glasses_reflection_score >= 70);
            AddResultRow("Glasses Frame", result.glasses_frame_score, result.glasses_frame_score >= 70);
            AddResultRow("Glasses Covering", result.glasses_covering_score, result.glasses_covering_score >= 70);
            AddResultRow("Hat", result.hat_score, result.hat_score >= 70);
            AddResultRow("Veil", result.veil_score, result.veil_score >= 70);
            AddResultRow("Skin Color", result.skin_color_score, result.skin_color_score >= 70);
            AddResultRow("Red Eyes", result.red_eyes_score, result.red_eyes_score >= 70);
            AddResultRow("Skin Reflection", result.skin_reflection_score, result.skin_reflection_score >= 70);
            AddResultRow("Shadow Face", result.shadow_face_score, result.shadow_face_score >= 70);
            AddResultRow("Shadow Background", result.shadow_background_score, result.shadow_background_score >= 70);
            AddResultRow("Background Uniformity", result.background_uniformity_score, result.background_uniformity_score >= 70);
            AddResultRow("Ink Marks", result.ink_marks_score, result.ink_marks_score >= 70);
            AddResultRow("Other Faces", result.other_faces_score, result.other_faces_score >= 70);
            AddResultRow("Overall", result.overall_score, result.overall_score >= 70);
        }

        private void AddResultRow(string metric, float score, bool pass)
        {
            dataGridView.Rows.Add(
                metric,
                score.ToString("F2"),
                pass ? "Pass" : "Fail"
            );
        }

    }

    public class PassportPhotoResult
    {
        public string base64image { get; set; }
        public string base64image_face_detect { get; set; }
        public string base64image_head_detect { get; set; }
        public string base64image_bg_removed { get; set; }
        public string base64image_bg_removed_canny { get; set; }
        public string base64image_bg_removed_canny_upd_crown { get; set; }
        public float face_height_proportion { get; set; }
        public float face_width_proportion { get; set; }
        public float head_height_proportion { get; set; }
        public float head_width_proportion { get; set; }
        public float blur_score { get; set; }
        public float pixelation_score { get; set; }
        public float white_noise_score { get; set; }
        public float contrast_score { get; set; }
        public float general_illumination_score { get; set; }
        public float face_position_score { get; set; }
        public float face_pose_score { get; set; }
        public float expression_score { get; set; }
        public float eyes_open_score { get; set; }
        public float eyes_direction_score { get; set; }
        public float mouth_open_score { get; set; }
        public float hair_over_face_score { get; set; }
        public float sunglasses_score { get; set; }
        public float glasses_reflection_score { get; set; }
        public float glasses_frame_score { get; set; }
        public float glasses_covering_score { get; set; }
        public float hat_score { get; set; }
        public float veil_score { get; set; }
        public float skin_color_score { get; set; }
        public float red_eyes_score { get; set; }
        public float skin_reflection_score { get; set; }
        public float shadow_face_score { get; set; }
        public float shadow_background_score { get; set; }
        public float background_uniformity_score { get; set; }
        public float ink_marks_score { get; set; }
        public float other_faces_score { get; set; }
        public float overall_score { get; set; }
        public string resultcode { get; set; }
        public string resultmessage { get; set; }
    }
}