(TeX-add-style-hook
 "Project"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("babel" "USenglish") ("biblatex" "style=authoryear" "maxbibnames=30") ("cleveref" "capitalize" "noabbrev")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "project_440_540"
    "inputenc"
    "fontenc"
    "babel"
    "csquotes"
    "booktabs"
    "amsfonts"
    "nicefrac"
    "microtype"
    "xcolor"
    "hyperref"
    "url"
    "tikz"
    "float"
    "biblatex"
    "bibentry"
    "cleveref")
   (TeX-add-symbols
    "citet"
    "citep")
   (LaTeX-add-bibliographies
    "refs"))
 :latex)

