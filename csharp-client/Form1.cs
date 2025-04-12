using System;
using System.Data;
using System.Data.SQLite; // 或使用 Microsoft.Data.Sqlite
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;

namespace detectDebris
{
    public partial class Form1 : Form
    {
        private Timer updateTimer; // 定义定时器

        public Form1()
        {
            InitializeComponent();
            InitializeWebView();
            InitializeDataGridView(); // 初始化 DataGridView
            InitializeTimer(); // 初始化定时器
        }

        private async void InitializeWebView()
        {
            await webView21.EnsureCoreWebView2Async(null);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            // 加载摄像图像
            webView21.Source = new Uri("http://localhost:5000/video_feed");

            // 启动定时器
            updateTimer.Start();
        }

        private void InitializeTimer()
        {
            updateTimer = new Timer();
            updateTimer.Interval = 1000; // 每1000毫秒（1秒）触发一次
            updateTimer.Tick += UpdateTimer_Tick; // 绑定事件处理器
        }

        private void UpdateTimer_Tick(object sender, EventArgs e)
        {
            // 每秒更新垃圾统计数据
            DataTable stats = GetGarbageStatisticsFromDatabase();
            DisplayGarbageStatistics(stats);
        }

        private DataTable GetGarbageStatisticsFromDatabase()
        {
            DataTable dataTable = new DataTable();
            string connectionString = @"Data Source=F:\LjmuStudy\engineeringProject\DataBase\yolo.db;Version=3;";

            using (SQLiteConnection connection = new SQLiteConnection(connectionString))
            {
                connection.Open();
                string query = "SELECT type, count FROM GarbageStats"; // 替换为你的查询
                using (SQLiteCommand command = new SQLiteCommand(query, connection))
                {
                    using (SQLiteDataAdapter adapter = new SQLiteDataAdapter(command))
                    {
                        adapter.Fill(dataTable); // 将结果填充到 DataTable
                    }
                }
            }

            return dataTable;
        }

        private void DisplayGarbageStatistics(DataTable stats)
        {
            if (stats.Rows.Count > 0)
            {
                dataGridView1.DataSource = stats; // 将 DataTable 绑定到 DataGridView
            }
            else
            {
                //MessageBox.Show("没有找到垃圾统计数据。");
            }
        }

        private void InitializeDataGridView()
        {
            // 这里可以设置 DataGridView 的一些属性，比如列标题等
            dataGridView1.Columns.Clear();
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            // 确保在窗体关闭时停止定时器
            updateTimer.Stop();
            updateTimer.Dispose();
        }
        private void toolStripMenuItem1_Click(object sender, EventArgs e)
        {
            DataSearch DataSearchform = new DataSearch();
            DataSearchform.Show();
        }

        private void dataAnalysisToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DataAnalysis DataAnalysisform = new DataAnalysis();
            DataAnalysisform.Show();
        }
    }
}
