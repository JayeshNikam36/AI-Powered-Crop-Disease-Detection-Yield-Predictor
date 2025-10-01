import { Activity, Cpu, HardDrive, Clock, CheckCircle2, AlertCircle } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const cpuData = [
  { time: "00:00", usage: 45 },
  { time: "04:00", usage: 38 },
  { time: "08:00", usage: 62 },
  { time: "12:00", usage: 78 },
  { time: "16:00", usage: 85 },
  { time: "20:00", usage: 52 },
];

const memoryData = [
  { time: "00:00", usage: 3.2 },
  { time: "04:00", usage: 3.4 },
  { time: "08:00", usage: 4.1 },
  { time: "12:00", usage: 5.3 },
  { time: "16:00", usage: 5.8 },
  { time: "20:00", usage: 4.5 },
];

export default function Monitor() {
  const services = [
    { name: "API Server", status: "operational", uptime: "99.98%", latency: "45ms" },
    { name: "Model Inference", status: "operational", uptime: "99.95%", latency: "320ms" },
    { name: "Database", status: "operational", uptime: "99.99%", latency: "12ms" },
    { name: "Storage", status: "operational", uptime: "100%", latency: "8ms" },
  ];

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">System Monitor</h1>
        <p className="text-muted-foreground">Real-time system health and performance metrics</p>
      </div>

      {/* System Status */}
      <Card className="bg-gradient-primary border-0 text-primary-foreground">
        <CardContent className="flex items-center gap-4 p-6">
          <div className="h-12 w-12 rounded-full bg-white/20 flex items-center justify-center">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold">All Systems Operational</h3>
            <p className="text-sm opacity-90">99.97% uptime • 0 incidents in last 30 days</p>
          </div>
          <Badge variant="outline" className="bg-white/20 border-white/40">
            Live
          </Badge>
        </CardContent>
      </Card>

      {/* Service Status */}
      <Card>
        <CardHeader>
          <CardTitle>Service Status</CardTitle>
          <CardDescription>Current status of all platform services</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {services.map((service) => (
              <div
                key={service.name}
                className="flex items-center justify-between p-4 bg-muted/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-success" />
                  <div>
                    <p className="font-medium">{service.name}</p>
                    <p className="text-sm text-muted-foreground">Uptime: {service.uptime}</p>
                  </div>
                </div>
                <div className="text-right">
                  <Badge variant="outline" className="bg-success/10 text-success border-success/20">
                    Operational
                  </Badge>
                  <p className="text-sm text-muted-foreground mt-1">Latency: {service.latency}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* CPU Usage */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>CPU Usage</CardTitle>
                <CardDescription>Server processor utilization</CardDescription>
              </div>
              <Cpu className="h-5 w-5 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-3xl font-bold">68%</span>
                <Badge variant="outline">Normal</Badge>
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={cpuData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="usage" stroke="hsl(var(--chart-1))" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Memory Usage */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Memory Usage</CardTitle>
                <CardDescription>RAM utilization (GB)</CardDescription>
              </div>
              <HardDrive className="h-5 w-5 text-primary" />
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-3xl font-bold">4.8 GB</span>
                <Badge variant="outline">8 GB Total</Badge>
              </div>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={memoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis domain={[0, 8]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="usage" stroke="hsl(var(--chart-2))" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Request Metrics */}
      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">API Requests</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12,847</div>
            <p className="text-xs text-muted-foreground">Last 24 hours</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">142ms</div>
            <p className="text-xs text-muted-foreground">↓ 12% from yesterday</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">0.03%</div>
            <p className="text-xs text-muted-foreground">Well below threshold</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
