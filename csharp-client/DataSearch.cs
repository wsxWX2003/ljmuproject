using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SQLite;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace detectDebris
{
    public partial class DataSearch : Form
    {
        public DataSearch()
        {
            InitializeComponent();
            dataGridView1.AllowUserToAddRows = false;
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

                dataGridView1.Columns.Clear();
                dataGridView1.Rows.Clear();

                dataGridView1.Columns.Add("type", "Type");
                dataGridView1.Columns.Add("total_count", "Total Count");

                while (reader.Read())
                {
                    dataGridView1.Rows.Add(reader["type"], reader["total_count"]);
                }
            }

            // 查询完成后，自动绘制柱状图
            //ShowChart();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string connString = "Data Source=F:\\LjmuStudy\\engineeringProject\\DataBase\\yolo.db;Version=3;";
            string typeQuery = comboBox1.Text.Trim();  // 从文本框获取用户输入的垃圾类型

            if (string.IsNullOrEmpty(typeQuery))
            {
                MessageBox.Show("请输入垃圾类型进行查询");
                return;
            }

            using (SQLiteConnection conn = new SQLiteConnection(connString))
            {
                conn.Open();
                // SQL 查询：根据垃圾类型查询所有的 type 和 timestamp
                string sql = "SELECT type, timestamp " +
                             "FROM GarbageStats " +
                             "WHERE type = @type " +
                             "ORDER BY timestamp DESC";  // 按时间戳降序排列

                SQLiteCommand cmd = new SQLiteCommand(sql, conn);
                cmd.Parameters.AddWithValue("@type", typeQuery);  // 绑定参数

                SQLiteDataReader reader = cmd.ExecuteReader();

                // 确保DataGridView有列
                dataGridView1.Columns.Clear(); // 清空旧列
                dataGridView1.Rows.Clear();    // 清空旧数据

                dataGridView1.Columns.Add("type", "Type");
                dataGridView1.Columns.Add("timestamp", "Timestamp");


                dataGridView1.Rows.Clear(); // 清空原数据

                // 读取每一条数据并显示在 DataGridView 中
                while (reader.Read())
                {
                    dataGridView1.Rows.Add(reader["type"], reader["timestamp"]);

                }
            }
        }

        private void LoadGarbageTypes()
        {
            // 定义垃圾类型
            string[] garbageTypes = new string[]
            {
        "bottle",
        "plastic",
            };

            // 清空ComboBox，并添加新的数据
            comboBox1.Items.Clear();
            comboBox1.Items.AddRange(garbageTypes);
        }
    }
}
