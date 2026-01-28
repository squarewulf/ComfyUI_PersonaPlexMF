import { FC } from "react";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement>; 
export const Button: FC<ButtonProps> = ({ children, className, ...props }) => {
  return (
    <button
      className={`rounded-lg shadow-sm disabled:bg-gray-100 bg-white hover:text-black py-2 px-3 active:text-black ${className ?? ""}`}
      {...props}
    >
      {children}
    </button>
  );
};
