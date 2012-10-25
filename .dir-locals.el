;; DOLFIN code indentation and timestamping for emacs.
;;
;; You may want to add the following to your .emacs to enable automatic
;; timestamping (only enabled when time-stamp-active is true):
;;
;;(add-hook 'before-save-hook 'time-stamp)

(
 ;; C++ mode
 (c++-mode . (
	      (c-basic-offset        . 2)
	      (indent-tabs-mode      . nil)
	      (c-file-offsets        . (
				        (substatement-open . 0)
					(brace-list-open   . 0)
					))
	      (time-stamp-start      . "Last changed: "); start of pattern
	      (time-stamp-end        . "\n")            ; end of pattern
	      (time-stamp-active     . t)               ; do enable time-stamps
	      (time-stamp-line-limit . 30)              ; check first 20 lines
	      (time-stamp-format     . "%04y-%02m-%02d"); date format
	      ))

 ;; C mode -- used for .h files by default
 (c-mode . (
	    (mode                  . c++)             ; switch to c++ mode
	    ;; The remainder is a copy of c++-mode
	    (c-basic-offset        . 2)
	    (indent-tabs-mode      . nil)
	    (c-file-offsets        . (
				      (substatement-open . 0)
				      (brace-list-open   . 0)
				      ))
	    (time-stamp-start      . "Last changed: "); start of pattern
	    (time-stamp-end        . "\n")            ; end of pattern
	    (time-stamp-active     . t)               ; do enable time-stamps
	    (time-stamp-line-limit . 30)              ; check first 20 lines
	    (time-stamp-format     . "%04y-%02m-%02d"); date format
	    ))

;; Python mode
 (python-mode . (
		 (time-stamp-start      . "Last changed: "); start of pattern
		 (time-stamp-end        . "\n")            ; end of pattern
		 (time-stamp-active     . t)               ; do enable time-stamps
		 (time-stamp-line-limit . 30)              ; check first 20 lines
		 (time-stamp-format     . "%04y-%02m-%02d"); date format
		 ))
 )
