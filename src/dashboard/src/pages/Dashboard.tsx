import { Activity, Image, TrendingUp, Leaf, CheckCircle2 } from "lucide-react";
import { MetricCard } from "@/components/MetricCard";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome to CropAI - AI-Powered Crop Disease Detection & Yield Prediction Platform
        </p>
      </div>

      {/* System Status Banner */}
      <Card className="bg-gradient-primary border-0 text-primary-foreground">
        <CardContent className="flex items-center gap-4 p-6">
          <div className="h-12 w-12 rounded-full bg-white/20 flex items-center justify-center">
            <CheckCircle2 className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">System Status: Operational</h3>
            <p className="text-sm opacity-90">All services running smoothly • Last updated: Just now</p>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Images Analyzed"
          value="12,847"
          description="Total samples processed"
          icon={Image}
          trend={{ value: 12.5, isPositive: true }}
        />
        <MetricCard
          title="Detection Accuracy"
          value="94.2%"
          description="Average model confidence"
          icon={Activity}
          trend={{ value: 2.1, isPositive: true }}
        />
        <MetricCard
          title="Active Alerts"
          value="8"
          description="Requires attention"
          icon={Leaf}
        />
        <MetricCard
          title="Yield Forecast"
          value="+18%"
          description="Predicted improvement"
          icon={TrendingUp}
          trend={{ value: 5.2, isPositive: true }}
        />
      </div>

      {/* Recent Activity & Quick Stats */}
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Most Common Disease</CardTitle>
            <CardDescription>Top detected in last 30 days</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-lg">Late Blight</p>
                  <p className="text-sm text-muted-foreground">Phytophthora infestans</p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">342</p>
                  <p className="text-xs text-muted-foreground">detections</p>
                </div>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full" style={{ width: "68%" }} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Get started with key features</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <a
              href="/detection"
              className="flex items-center gap-3 p-3 rounded-lg bg-secondary hover:bg-secondary/80 transition-colors"
            >
              <Activity className="h-5 w-5 text-primary" />
              <div>
                <p className="font-medium">Analyze New Sample</p>
                <p className="text-xs text-muted-foreground">Upload leaf image for detection</p>
              </div>
            </a>
            <a
              href="/knowledge"
              className="flex items-center gap-3 p-3 rounded-lg bg-secondary hover:bg-secondary/80 transition-colors"
            >
              <Leaf className="h-5 w-5 text-primary" />
              <div>
                <p className="font-medium">Browse Knowledge Base</p>
                <p className="text-xs text-muted-foreground">Learn about diseases & treatments</p>
              </div>
            </a>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
