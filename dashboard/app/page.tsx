"use client";

import { useState } from "react";
import BatchRuns from "@/components/BatchRuns";
import ScoreDistribution from "@/components/ScoreDistribution";
import Benchmark from "@/components/Benchmark";
import CustomerLookup from "@/components/CustomerLookup";

const TABS = [
  { id: "runs", label: "Batch Run History", el: <BatchRuns /> },
  { id: "dist", label: "Score Distribution", el: <ScoreDistribution /> },
  { id: "bench", label: "Benchmark", el: <Benchmark /> },
  { id: "lookup", label: "Customer Lookup", el: <CustomerLookup /> },
] as const;

export default function Page() {
  const [active, setActive] = useState<string>("runs");
  return (
    <div>
      <nav className="mb-6 flex flex-wrap gap-2 border-b" style={{ borderColor: "var(--border)" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm font-medium transition ${
              active === t.id
                ? "border-brand text-brand"
                : "border-transparent hover:opacity-80"
            }`}
            style={active === t.id ? { color: "#4f46e5", borderColor: "#4f46e5" } : {}}
          >
            {t.label}
          </button>
        ))}
      </nav>
      {TABS.map((t) => (
        <div key={t.id} hidden={active !== t.id}>
          {t.el}
        </div>
      ))}
    </div>
  );
}
