#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

// WiFi 热点配置
const char* ssid = "esp32cam";
const char* password = "12345678";

// 视频流处理函数
esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char header_buf[64];

  httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");

  while (true) {
    // 如果客户端断开了连接，就退出
    if (!req->handle) {
      break;
    }

    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      continue;
    }

    size_t header_len = snprintf(header_buf, sizeof(header_buf),
                                 "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);

    res = httpd_resp_send_chunk(req, header_buf, header_len);
    if (res != ESP_OK) {
      esp_camera_fb_return(fb);
      break;
    }

    res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;

    res = httpd_resp_send_chunk(req, "\r\n", 2);
    if (res != ESP_OK) break;

    vTaskDelay(pdMS_TO_TICKS(50)); // 控制帧率 ~20fps
  }

  return res;
}

// 启动视频流服务器
void startCameraServer() {
  httpd_handle_t stream_httpd = NULL;
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// 设置函数
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // 摄像头初始化配置
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_HD;     // 1280x720，高清流
  config.jpeg_quality = 8;              // 画质较高（0=最好，63=最差）
  config.fb_count = 2;                  // 双缓冲，防止卡顿
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // 初始化摄像头
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }

  // 启动WiFi热点
  WiFi.softAP(ssid, password);
  WiFi.setSleep(false);  // 关闭省电，保证持续传输性能

  Serial.println();
  Serial.print("WiFi 热点已启动，名称: ");
  Serial.println(ssid);
  Serial.print("访问视频流: http://");
  Serial.print(WiFi.softAPIP());
  Serial.println("/stream");

  // 启动视频服务器
  startCameraServer();
}

void loop() {
  delay(1000);
}

