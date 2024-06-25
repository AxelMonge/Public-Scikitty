import Link from 'next/link';
import Image from 'next/image';

const ToolCard = ({ name, image, url }) => {
    return (
        <div className="text-black text-center min-w-[300px]">
            <Link href={url} replace>
                <Image
                    width={3840}
                    height={2160}
                    src={image}
                    alt={name}
                    className="h-[50vh] rounded-t w-full object-cover w-[350px] bg-[#323C4F]"
                />
            </Link>
            <Link href={url} replace>
                <h5 className="py-1 hover:text-[#777] bg-[#D9D9D9] rounded-b">
                    {name}
                </h5>
            </Link>
        </div>
    )
};

export default ToolCard;