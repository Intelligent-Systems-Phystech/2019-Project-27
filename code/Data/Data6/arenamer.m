constanta = -300;
numdocs = 17;
beep
%{
for ii=319:318+numdocs
    movefile(['lengthForTerms',num2str(ii),'.txt'],['lengthForTerms',num2str(ii+constanta),'.txt']);
end
%}

for ii=301:300+numdocs
    movefile(['termvars',num2str(ii),'.txt'],['termvars',num2str(ii+constanta),'.txt']);
end
for ii=301:300+numdocs
    movefile(['doc_id_strings',num2str(ii),'.txt'],['doc_id_strings',num2str(ii+constanta),'.txt']);
end
