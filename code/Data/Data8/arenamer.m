constanta = -400;
numdocs = 50;
beep
%{
for ii=401:400+numdocs
    movefile(['lengthForTerms',num2str(ii),'.txt'],['lengthForTerms',num2str(ii+constanta),'.txt']);
end
%}
beep
for ii=401:400+numdocs
    movefile(['termvars',num2str(ii),'.txt'],['termvars',num2str(ii+constanta),'.txt']);
end
for ii=401:400+numdocs
    movefile(['doc_id_strings',num2str(ii),'.txt'],['doc_id_strings',num2str(ii+constanta),'.txt']);
end