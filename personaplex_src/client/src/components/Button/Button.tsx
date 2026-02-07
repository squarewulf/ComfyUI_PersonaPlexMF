import { FC } from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost';
};

export const Button: FC<ButtonProps> = ({ children, className, variant = 'primary', ...props }) => {
  const base = "font-medium rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[#76b900]/40 disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-[#76b900] hover:bg-[#8ad400] text-black py-2.5 px-6 shadow-lg shadow-[#76b900]/20 hover:shadow-[#76b900]/30 active:scale-[0.98]",
    secondary: "bg-white/5 hover:bg-white/10 text-white/80 py-2 px-4 border border-white/10",
    ghost: "bg-transparent hover:bg-white/5 text-white/60 hover:text-white/80 py-1.5 px-3",
  };
  return (
    <button
      className={`${base} ${variants[variant]} ${className ?? ""}`}
      {...props}
    >
      {children}
    </button>
  );
};
