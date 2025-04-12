using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SQLite;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices.ComTypes;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace detectDebris
{
    public partial class DataAnalysis : Form
    {
        public DataAnalysis()
        {
            InitializeComponent();
            LoadGarbageTypes();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string connString = "Data Source=F:\\LjmuStudy\\engineeringProject\\DataBase\\yolo.db;Version=3;";
            using (SQLiteConnection conn = new SQLiteConnection(connString))
            {
                conn.Open();
                string sql = "SELECT type, COUNT(*) AS total_count " +
                             "FROM GarbageStats " +
                             "WHERE timestamp BETWEEN @start AND @end " +
                             "GROUP BY type " +
                             "ORDER BY total_count DESC";

                SQLiteCommand cmd = new SQLiteCommand(sql, conn);
                cmd.Parameters.AddWithValue("@start", dateTimePicker1.Value.ToString("yyyy-MM-dd 00:00:00"));
                cmd.Parameters.AddWithValue("@end", dateTimePicker2.Value.ToString("yyyy-MM-dd 23:59:59"));

                SQLiteDataReader reader = cmd.ExecuteReader();

                // 清空原有的Series数据
                chart1.Series.Clear();

                // 创建一个新的Series用于柱状图
                Series series = new Series("Garbage Types");
                series.ChartType = SeriesChartType.Column;  // 设置柱状图类型
                chart1.Series.Add(series);

                int maxCount = 0;

                // 读取数据并填充到柱状图
                while (reader.Read())
                {
                    string type = reader["type"].ToString();
                    int count = Convert.ToInt32(reader["total_count"]);

                    // 更新最大值
                    maxCount = Math.Max(maxCount, count);

                    // 向Series中添加数据
                    series.Points.AddXY(type, count);  // X轴为垃圾类型，Y轴为数量
                }

                // 设置 Y 轴的最大值，确保是整数且大于最大数据
                int yAxisMax = (int)Math.Ceiling(maxCount * 1.2);  // 增加一定的空间，例如增加 20%
                yAxisMax = yAxisMax + (yAxisMax % 2);  // 确保最大值为偶数（如果需要的话）

                // 设置 Y 轴的间隔为 1，显示为 1, 2, 3, 4, 5...
                chart1.ChartAreas[0].AxisY.Interval = 1;
                chart1.ChartAreas[0].AxisY.Maximum = yAxisMax;
                chart1.ChartAreas[0].AxisY.Minimum = 0;  // Y轴从0开始

                // 确保Y轴从0开始，且显示整数
                chart1.ChartAreas[0].AxisY.IsStartedFromZero = true;
                chart1.ChartAreas[0].AxisY.LabelStyle.Format = "#";  // 格式化为整数
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {

            chart1.ChartAreas[0].AxisX.Interval = 1;  // 按天显示
            chart1.ChartAreas[0].AxisX.IntervalType = DateTimeIntervalType.Days; // X 轴的单位是天
            chart1.ChartAreas[0].AxisX.LabelStyle.Format = "yyyy-MM-dd";  // 格式化日期
            chart1.ChartAreas[0].AxisX.LabelStyle.Angle = -45; // 旋转标签，防止重叠
            chart1.ChartAreas[0].AxisX.LabelStyle.IsEndLabelVisible = true; // 确保最后一个日期可见

            string connString = "Data Source=F:\\LjmuStudy\\engineeringProject\\DataBase\\yolo.db;Version=3;";
            string typeQuery = comboBox1.Text.Trim();  // 获取用户输入的垃圾类型

            if (string.IsNullOrEmpty(typeQuery))
            {
                MessageBox.Show("请输入垃圾类型进行查询");
                return;
            }

            using (SQLiteConnection conn = new SQLiteConnection(connString))
            {
                conn.Open();
                string sql = "SELECT DATE(timestamp) AS date, COUNT(*) AS total_count " +
                             "FROM GarbageStats " +
                             "WHERE type = @type " +
                             "GROUP BY DATE(timestamp) " +
                             "ORDER BY date ASC";

                SQLiteCommand cmd = new SQLiteCommand(sql, conn);
                cmd.Parameters.AddWithValue("@type", typeQuery);  // 绑定参数

                SQLiteDataReader reader = cmd.ExecuteReader();

                // 清空图表
                chart1.Series.Clear();

                // 添加折线图
                Series series = chart1.Series.Add(typeQuery);
                series.ChartType = SeriesChartType.Line;

                // 设置折线图的标记样式
                series.MarkerStyle = MarkerStyle.Circle;  // 数据点为圆形
                series.MarkerSize = 8;  // 设置标记大小
                series.MarkerColor = Color.Red;  // 设置标记颜色

                // 读取数据并填充到折线图
                while (reader.Read())
                {
                    DateTime date = Convert.ToDateTime(reader["date"]); // 只获取日期
                    int count = Convert.ToInt32(reader["total_count"]); // 每天的垃圾数量
                    series.Points.AddXY(date, count); // X 轴为日期，Y 轴为数量
                }
            }

        }
        private void LoadGarbageTypes()
        {
            // 定义垃圾类型
            string[] garbageTypes = new string[]
            {
        "plastic",
        "bottle",
            };

            // 清空ComboBox，并添加新的数据
            comboBox1.Items.Clear();
            comboBox1.Items.AddRange(garbageTypes);
        }
    }
}
