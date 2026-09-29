// FinHelm 的标：舵轮（helm）。自己画的，不用任何产品的标
export function Logo({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" className="logo" aria-hidden>
      <circle cx="12" cy="12" r="7.5" />
      <circle cx="12" cy="12" r="2.2" />
      <path d="M12 1.5v8.3M12 14.2v8.3M1.5 12h8.3M14.2 12h8.3M4.6 4.6l5.8 5.8M13.6 13.6l5.8 5.8M19.4 4.6l-5.8 5.8M10.4 13.6l-5.8 5.8" />
    </svg>
  );
}
