% Copyright (C) 2005 Anders Logg.
% Licensed under the GNU GPL Version 2.
%
% First added:  2004-09-03
% Last changed: 2005-09-30
%
% LaTeX document class for FEniCS manuals.

%--- Set up class ----
\ProvidesClass{fenicsmanual}[2005/09/03 FEniCS manual]
\NeedsTeXFormat{LaTeX2e}
\LoadClass[12pt,twoside]{book}

%--- Load packages ---
\RequirePackage{graphicx}
\RequirePackage{psfrag}
\RequirePackage{fancyhdr}
\RequirePackage{fancybox}
\RequirePackage{fancyvrb}
\RequirePackage{sectsty}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{makeidx}
\RequirePackage{url}
\RequirePackage[latin1]{inputenc}

%--- Misc options ---
\setlength{\parindent}{0pt}
\setlength{\parskip}{12pt}
\allsectionsfont{\sffamily}
\makeindex

%--- Remove header and footer from blank pages  ---
\let\origdoublepage\cleardoublepage
\newcommand{\clearemptydoublepage}{%
  \clearpage
  {\pagestyle{empty}\origdoublepage}%
}
\let\cleardoublepage\clearemptydoublepage

%--- Print index at end of document ---
\AtEndDocument{\cleardoublepage\printindex}

%--- Variables ---
\newcommand{\@fenicstitle}{}
\newcommand{\fenicstitle}[1]{\renewcommand{\@fenicstitle}{#1}}
\newcommand{\@fenicsauthor}{}
\newcommand{\fenicsauthor}[1]{\renewcommand{\@fenicsauthor}{#1}}
\newcommand{\@fenicsimage}{\vspace{8cm}}
\newcommand{\fenicsimage}[1]{\renewcommand{\@fenicsimage}{
    \begin{center}
      \includegraphics[height=8cm]{#1}
    \end{center}}}
\newcommand{\@fenicspackage}{<package unspecified>}
\newcommand{\@fenicspackagett}{<package unspecified>}
\newcommand{\fenicspackage}[2]{\renewcommand{\@fenicspackage}{#1}\renewcommand{\@fenicspackagett}{#2}}
\newcommand{\package}{\@fenicspackage}
\newcommand{\packagett}{\@fenicspackagett}

%--- Commands ---
\renewcommand{\maketitle}{
  \lhead{\textsf{\textbf{\@fenicstitle}}}
  \rhead{\textsf{\@fenicsauthor}}
  \pagestyle{fancy}
  \renewcommand{\footrulewidth}{2pt}
  \renewcommand{\headrulewidth}{2pt}
  \thispagestyle{empty}
  \Large\textsf{\textbf{\@fenicstitle}} \\
  \vspace{-0.5cm}
  \hrule height 2pt
  \hfill\large\textsf{\today}
  \vspace{3cm}
  \@fenicsimage
  \vfill\large\textsf{\textbf{\@fenicsauthor}} \\
  \hrule height 2pt
  \hfill\large\texttt{www.fenics.org}
  \newpage
  \null\vfill
  \normalsize
  Visit \texttt{http://www.fenics.org/} for the latest version of this manual. \\
  Send comments and suggestions to \texttt{\@fenicspackagett{}-dev@fenics.org}.
  \thispagestyle{empty}
  \cleardoublepage
  \tableofcontents}

\newcommand{\fenics}{\textbf{\textsf{\normalsize{FE}\Large{ni}\normalsize{CS}}}}
\newcommand{\dolfin}{\textbf{\textsf{DOLFIN}}}
\newcommand{\ffc}{\textbf{\textsf{FFC}}}
\newcommand{\fiat}{\textbf{\textsf{FIAT}}}
\newcommand{\fixme}[1]{\ \\ \begin{tabular}{||p{\textwidth}||}\hline\rm\textbf{FIXME:}\rm #1 \\ \hline\end{tabular} \\}
\newcommand{\devnote}[1]{$\blacktriangleright$ \emph{Developer's note:} #1}

%--- Environments ---
\DefineVerbatimEnvironment{code}{Verbatim}{commandchars=\\\{\}}

%--- Macros ---
\newcommand{\dx}{\, \mathrm{d}x}
\newcommand{\dX}{\, \mathrm{d}X}
\newcommand{\R}{\mathbb{R}}
