const root = document.documentElement;
const button = document.getElementById("theme");
const saved = localStorage.getItem("studymate-theme");
if (saved) root.dataset.theme = saved;
button.addEventListener("click", () => {
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  localStorage.setItem("studymate-theme", next);
});

// Плавное появление секций
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      entry.target.style.opacity = "1";
      entry.target.style.transform = "translateY(0)";
    }
  }
}, { threshold: 0.08 });
document.querySelectorAll(".section, .features article, .shots figure").forEach((el) => {
  el.style.opacity = "0";
  el.style.transform = "translateY(14px)";
  el.style.transition = "opacity .5s ease, transform .5s ease";
  observer.observe(el);
});
