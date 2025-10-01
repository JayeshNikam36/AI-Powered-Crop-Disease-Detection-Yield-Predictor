import { Search, Leaf, AlertTriangle, Shield } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

const diseases = [
  {
    name: "Late Blight",
    scientificName: "Phytophthora infestans",
    severity: "High",
    symptoms: "Dark brown spots on leaves, white fungal growth on undersides",
    causes: "Fungal infection in cool, wet conditions",
    prevention: "Remove infected plants, apply fungicides, ensure good air circulation",
  },
  {
    name: "Early Blight",
    scientificName: "Alternaria solani",
    severity: "Medium",
    symptoms: "Concentric rings on older leaves, yellowing and browning",
    causes: "Fungal pathogen thriving in warm, humid conditions",
    prevention: "Crop rotation, avoid overhead watering, apply copper-based fungicides",
  },
  {
    name: "Septoria Leaf Spot",
    scientificName: "Septoria lycopersici",
    severity: "Medium",
    symptoms: "Small circular spots with gray centers on lower leaves",
    causes: "Fungal infection spread by rain splash",
    prevention: "Mulch around plants, remove infected leaves, use resistant varieties",
  },
  {
    name: "Bacterial Spot",
    scientificName: "Xanthomonas spp.",
    severity: "High",
    symptoms: "Small dark spots with yellow halos on leaves and fruit",
    causes: "Bacterial infection in warm, wet conditions",
    prevention: "Use disease-free seeds, copper sprays, avoid working with wet plants",
  },
];

export default function Knowledge() {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "High":
        return "bg-destructive text-destructive-foreground";
      case "Medium":
        return "bg-warning text-warning-foreground";
      default:
        return "bg-success text-success-foreground";
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Knowledge Base</h1>
        <p className="text-muted-foreground">
          Comprehensive database of crop diseases, symptoms, and treatment recommendations
        </p>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <Input placeholder="Search diseases, symptoms, or treatments..." className="pl-10" />
          </div>
        </CardContent>
      </Card>

      {/* Disease Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        {diseases.map((disease) => (
          <Card key={disease.name} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="space-y-1">
                  <CardTitle className="text-xl">{disease.name}</CardTitle>
                  <CardDescription className="italic">{disease.scientificName}</CardDescription>
                </div>
                <Badge className={getSeverityColor(disease.severity)}>{disease.severity}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Leaf className="h-4 w-4 text-primary" />
                  Symptoms
                </div>
                <p className="text-sm text-muted-foreground pl-6">{disease.symptoms}</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <AlertTriangle className="h-4 w-4 text-warning" />
                  Causes
                </div>
                <p className="text-sm text-muted-foreground pl-6">{disease.causes}</p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <Shield className="h-4 w-4 text-success" />
                  Prevention & Treatment
                </div>
                <p className="text-sm text-muted-foreground pl-6">{disease.prevention}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
