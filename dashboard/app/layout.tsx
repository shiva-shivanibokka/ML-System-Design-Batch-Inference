import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Batch Inference Monitor",
  description:
    "Monitoring dashboard for a nightly PySpark batch churn-scoring pipeline (KKBox).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b" style={{ borderColor: "var(--border)" }}>
          <div className="mx-auto max-w-6xl px-4 py-4">
            <h1 className="text-lg font-bold">Batch Inference Monitor</h1>
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              Nightly churn scoring · PySpark → PostgreSQL → FastAPI · KKBox dataset
            </p>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
