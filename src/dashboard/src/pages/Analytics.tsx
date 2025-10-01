import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const accuracyData = [
  { name: "Jan", accuracy: 89.2, precision: 87.5, recall: 90.1 },
  { name: "Feb", accuracy: 90.5, precision: 88.9, recall: 91.2 },
  { name: "Mar", accuracy: 92.1, precision: 90.3, recall: 92.8 },
  { name: "Apr", accuracy: 93.4, precision: 91.7, recall: 93.9 },
  { name: "May", accuracy: 94.2, precision: 92.5, recall: 94.5 },
];

const diseaseDistribution = [
  { name: "Late Blight", value: 342, color: "hsl(var(--chart-1))" },
  { name: "Early Blight", value: 287, color: "hsl(var(--chart-2))" },
  { name: "Septoria Spot", value: 198, color: "hsl(var(--chart-3))" },
  { name: "Bacterial Spot", value: 156, color: "hsl(var(--chart-4))" },
  { name: "Healthy", value: 423, color: "hsl(var(--chart-5))" },
];

const detectionTrend = [
  { month: "Jan", detections: 1240 },
  { month: "Feb", detections: 1580 },
  { month: "Mar", detections: 1820 },
  { month: "Apr", detections: 2100 },
  { month: "May", detections: 2450 },
];

export default function Analytics() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Analytics</h1>
        <p className="text-muted-foreground">Model performance metrics and detection insights</p>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-6 md:grid-cols-4">
        {[
          { label: "Accuracy", value: "94.2%", color: "text-chart-1" },
          { label: "Precision", value: "92.5%", color: "text-chart-2" },
          { label: "Recall", value: "94.5%", color: "text-chart-3" },
          { label: "F1-Score", value: "93.5%", color: "text-chart-4" },
        ].map((metric) => (
          <Card key={metric.label}>
            <CardHeader className="pb-2">
              <CardDescription>{metric.label}</CardDescription>
            </CardHeader>
            <CardContent>
              <p className={`text-3xl font-bold ${metric.color}`}>{metric.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Performance Metrics Over Time */}
        <Card className="col-span-full">
          <CardHeader>
            <CardTitle>Model Performance Over Time</CardTitle>
            <CardDescription>Accuracy, precision, and recall trends</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[85, 100]} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="hsl(var(--chart-1))"
                  strokeWidth={2}
                  name="Accuracy"
                />
                <Line
                  type="monotone"
                  dataKey="precision"
                  stroke="hsl(var(--chart-2))"
                  strokeWidth={2}
                  name="Precision"
                />
                <Line
                  type="monotone"
                  dataKey="recall"
                  stroke="hsl(var(--chart-3))"
                  strokeWidth={2}
                  name="Recall"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Disease Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Disease Distribution</CardTitle>
            <CardDescription>Breakdown of detected diseases</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={diseaseDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry: any) => `${entry.name}: ${((entry.value / diseaseDistribution.reduce((a, b) => a + b.value, 0)) * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {diseaseDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Detection Trend */}
        <Card>
          <CardHeader>
            <CardTitle>Detection Trend</CardTitle>
            <CardDescription>Monthly detection volume</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={detectionTrend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="detections" fill="hsl(var(--chart-1))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
