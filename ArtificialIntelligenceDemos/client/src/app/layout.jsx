import { Karla } from "next/font/google";
import NavBar from "./components/NavBar";
import "@/app/styles/globals.css";
import { Toaster } from "sonner";

const inter = Karla({ subsets: ["latin"] });

export const metadata = {
  title: "Artificial Inteligence Demos",
  description: "Some Artificial Inteligence Demos to Learn",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className + " bg-page-color pageBackground"}>
        <Toaster richColors closeButton />
        <NavBar />
        {children}
      </body>
    </html>
  );
}
