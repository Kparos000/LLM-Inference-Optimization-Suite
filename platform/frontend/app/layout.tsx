import type { Metadata } from "next";
import "./globals.css";
import { ExperimentSessionProvider } from "@/lib/session";

export const metadata: Metadata = {
  title: "AI Inference Engineering Platform",
  description: "Guided replay of a full LLM inference optimization experiment."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ExperimentSessionProvider>{children}</ExperimentSessionProvider>
      </body>
    </html>
  );
}

