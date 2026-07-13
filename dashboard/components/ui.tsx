// Small shared presentational bits so every tab looks like one system.
import { ReactNode } from "react";

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <div
      className="rounded-xl border p-4 shadow-sm"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      {title && <h3 className="mb-3 text-sm font-semibold">{title}</h3>}
      {children}
    </div>
  );
}

export function StatTile({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Card>
      <div className="text-xs uppercase tracking-wide" style={{ color: "var(--muted)" }}>
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
    </Card>
  );
}

const STATUS_STYLE: Record<string, string> = {
  validated: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  completed: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  running: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
};

export function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_STYLE[status] ?? "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{status}</span>;
}

export function RiskBadge({ tier }: { tier: string }) {
  const t = tier.toLowerCase();
  const cls =
    t === "high"
      ? "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"
      : t === "medium"
      ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
      : "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300";
  return <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${cls}`}>{tier.toUpperCase()}</span>;
}

export function Empty({ msg }: { msg: string }) {
  return (
    <div className="py-10 text-center text-sm" style={{ color: "var(--muted)" }}>
      {msg}
    </div>
  );
}

export function fmt(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}
