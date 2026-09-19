import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import CreateProfile from "./pages/CreateProfile";
import Processing from "./pages/Processing";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/create" element={<CreateProfile />} />
      <Route path="/profiles/:id" element={<Processing />} />
    </Routes>
  );
}
