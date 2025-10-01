import { Plane as DroneIcon, MapPin, TrendingDown } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function Drone() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Drone Monitoring</h1>
        <p className="text-muted-foreground">
          Aerial crop health analysis with automated disease detection across fields
        </p>
      </div>

      <Alert className="bg-primary/10 border-primary/20">
        <DroneIcon className="h-4 w-4 text-primary" />
        <AlertTitle>Coming Soon</AlertTitle>
        <AlertDescription>
          Drone monitoring features are currently under development. This will enable large-scale field
          analysis with bounding box detection of infected areas.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Field Coverage</CardTitle>
            <CardDescription>Total area monitored</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-primary">0 ha</div>
            <p className="text-xs text-muted-foreground mt-1">Hectares analyzed</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Detected Issues</CardTitle>
            <CardDescription>Infected zones identified</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-warning">0</div>
            <p className="text-xs text-muted-foreground mt-1">Areas requiring attention</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Health Score</CardTitle>
            <CardDescription>Overall crop health</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-success">N/A</div>
            <p className="text-xs text-muted-foreground mt-1">Awaiting first scan</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Field Map Preview</CardTitle>
          <CardDescription>Aerial view with disease detection overlay</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
            <div className="text-center text-muted-foreground">
              <DroneIcon className="h-16 w-16 mx-auto mb-4" />
              <p className="text-lg font-semibold mb-2">No Drone Data Available</p>
              <p className="text-sm">Upload aerial imagery to begin field analysis</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Planned Features</CardTitle>
          <CardDescription>What's coming to drone monitoring</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="flex gap-3 p-4 bg-muted/50 rounded-lg">
              <MapPin className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold mb-1">GPS-Tagged Detection</h4>
                <p className="text-sm text-muted-foreground">
                  Precise location mapping of diseased areas with coordinates
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-muted/50 rounded-lg">
              <TrendingDown className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold mb-1">Infection Spread Tracking</h4>
                <p className="text-sm text-muted-foreground">
                  Monitor disease progression over time with historical comparisons
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-muted/50 rounded-lg">
              <Badge className="h-5 bg-primary shrink-0 mt-0.5">AI</Badge>
              <div>
                <h4 className="font-semibold mb-1">Automated Bounding Boxes</h4>
                <p className="text-sm text-muted-foreground">
                  AI-powered detection with visual overlays on infected zones
                </p>
              </div>
            </div>

            <div className="flex gap-3 p-4 bg-muted/50 rounded-lg">
              <DroneIcon className="h-5 w-5 text-primary shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold mb-1">Multi-Spectral Analysis</h4>
                <p className="text-sm text-muted-foreground">
                  NDVI and thermal imaging for early stress detection
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
