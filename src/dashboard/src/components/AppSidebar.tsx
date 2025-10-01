import { Home, Scan, Brain, BookOpen, Plane, BarChart3, Activity } from "lucide-react";
import { NavLink } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const menuItems = [
  { title: "Dashboard", url: "/", icon: Home },
  { title: "Disease Detection", url: "/detection", icon: Scan },
  { title: "Grad-CAM Analysis", url: "/gradcam", icon: Brain },
  { title: "Knowledge Base", url: "/knowledge", icon: BookOpen },
  { title: "Drone Monitoring", url: "/drone", icon: Plane },
  { title: "Analytics", url: "/analytics", icon: BarChart3 },
  { title: "System Monitor", url: "/monitor", icon: Activity },
];

export function AppSidebar() {
  const { open } = useSidebar();

  return (
    <Sidebar collapsible="icon">
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-base font-semibold px-4 py-3">
            {open && "CropAI Platform"}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild tooltip={item.title}>
                    <NavLink
                      to={item.url}
                      end={item.url === "/"}
                      className={({ isActive }) =>
                        `flex items-center gap-3 px-4 py-2.5 rounded-lg transition-all ${
                          isActive
                            ? "bg-primary text-primary-foreground shadow-md"
                            : "hover:bg-sidebar-accent"
                        }`
                      }
                    >
                      <item.icon className="h-5 w-5 shrink-0" />
                      {open && <span className="font-medium">{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
